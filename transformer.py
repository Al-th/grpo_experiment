import torch
import numpy
import gc
from torch.nn import functional as F

class SelfAttentionHead(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        
        
        self.individual_head_size = params.individual_head_size
        self.block_size = params.block_size
        
        self.key = torch.nn.Linear(params.n_embedding, self.individual_head_size, bias=False)
        self.query = torch.nn.Linear(params.n_embedding, self.individual_head_size, bias=False)
        self.value = torch.nn.Linear(params.n_embedding, self.individual_head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.register_buffer('k_cache', torch.zeros(0))
        self.register_buffer('v_cache', torch.zeros(0))
        
        self.use_cache = False
        self.reset_cache()
        

    def train(self, mode=True):
        super().train(mode)
        self.use_cache = not mode
        self.reset_cache()

    def eval(self):
        super().eval()
        self.use_cache = True
        self.reset_cache()
        
        
    def reset_cache(self):
        self.last_index = self.last_index = -1
        self.k_cache = torch.zeros(0, device=self.key.weight.device)
        self.v_cache = torch.zeros(0, device=self.value.weight.device)
        torch.cuda.empty_cache()


    def forward(self, x):
        B, T, _ = x.shape

        if self.use_cache:
            x_new = x[:,-1,:]
            if(self.k_cache.shape[0] == 0 and self.v_cache.shape[0] == 0):
                self.k_cache = torch.zeros(size=[B,self.block_size,self.individual_head_size], device=self.key.weight.device)
                self.v_cache = torch.zeros(size=[B,self.block_size,self.individual_head_size], device=self.value.weight.device)

            k_new = self.key(x_new) #batch_size, 1, individual_head_size
            q_new = self.query(x_new) # batch_size, 1, individual_head_size
            v_new = self.value(x_new) # batch_size, 1, individual_head_size

            self.last_index += 1
            
            update_index = self.last_index % self.block_size

            self.k_cache[:,update_index,:] = k_new
            self.v_cache[:,update_index,:] = v_new


            #Retrieve appropriate K, V by fetching the KV cache
            valid_start = max(0,self.last_index-self.block_size+1)
            cache_indices = torch.arange(valid_start, self.last_index+1, device=self.k_cache.device) % self.block_size 

            K = self.k_cache[:, cache_indices, :]
            V = self.v_cache[:, cache_indices, :]

            QKt = (q_new @ K.transpose(-1,-2)) * self.individual_head_size**-0.5
            
            QKt[:,T:,:] = float('-inf')
            wei = F.softmax(QKt, dim=-1)


            out = wei @ V
            return out
        else:
            k = self.key(x) # batch_size, block_size, individual_head_size
            q = self.query(x) # batch_size, block_size, individual_head_size
            v = self.value(x) # batch_size, block_size, individual_head_size
            
            QKt = (q @ k.transpose(-1, -2)) * (self.individual_head_size**-0.5)
            
            wei = QKt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1) 

            out = wei @ v
            return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_heads = params.n_heads

        self.heads = torch.nn.ModuleList([SelfAttentionHead(params) for _ in range(self.n_heads)])
        self.layer_norm = torch.nn.LayerNorm(params.n_embedding)
    
    def forward(self, x):
        x_handle = x
        out = x + torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.layer_norm(out)
        return out
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        
        
        
        self.ma_heads = MultiHeadAttention(params) 
        self.ffw = torch.nn.Sequential(
            torch.nn.Linear(params.n_embedding, 4*params.n_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(4*params.n_embedding, params.n_embedding),
            torch.nn.Dropout(0.2))
        self.layer_norm = torch.nn.LayerNorm(params.n_embedding)
        self.layer_norm2 = torch.nn.LayerNorm(params.n_embedding)
        
    def forward(self, x):
        #Apply the multi-head attention
        x = x + self.ma_heads(self.layer_norm(x))

        #Apply the feed forward layer     
        x = x + self.ffw(self.layer_norm2(x))
        return x
        
        

class DecoderTrans (torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.token_embedding = torch.nn.Embedding(params.vocab_size, params.n_embedding)  
        self.pos_embedding = torch.nn.Embedding(params.block_size, params.n_embedding)
        self.decoder_block = torch.nn.Sequential(*[DecoderBlock(params) for _ in range(params.n_decoder_blocks)])
        self.lln = torch.nn.LayerNorm(params.n_embedding)
        self.linear_layer = torch.nn.Linear(params.n_embedding, params.vocab_size)

    # Forward function can now be used with DataParallel wrapper
    def forward(self, fun_name, **kwargs):
        if(fun_name == 'forward'):
            return self.fwd(**kwargs)
        elif(fun_name == 'generate'):
            return self.generate(**kwargs)
        elif(fun_name == 'get_token_logprob'):
            return self.get_token_logprob(**kwargs)
        
    def fwd(self, x, y = None):
        B, T = x.shape # Size (batch_size, block_size)

        x = self.token_embedding(x) # Size (batch_size, block_size, n_embedding)
        
        # Absolute positional encoding is added to the token_embedding as per vanilla transformer
        positional_embedding = self.pos_embedding(torch.arange(0, T,  device=self.pos_embedding.weight.device))  # Size (block_size, n_embedding)
        x = x + positional_embedding # Size (batch_size, block_size, n_embedding)
        
        x = self.decoder_block(x) # Size (batch_size, block_size, n_embedding)
        x = self.lln(x) # Size (batch_size, block_size, n_embedding)
        logits = self.linear_layer(x) # Size (batch_size, block_size, vocab_size)
        
        if y is not None:
            # Flatten the logits across batch and token
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
        return logits, loss
    
    def generate(self, context, nb_tokens, callback=None):
        token_probs = None
        for i in range(nb_tokens):
            logits, loss  = self('forward', x=context[:,-self.params.block_size:])
            logits = logits[:,-1,:] # Take the last token logits for each batch
            probs = F.softmax(logits, dim=-1) # Compute the probabilities on the logits dimension )    
            logprobs = F.log_softmax(logits, dim=-1)
            generated_token = torch.multinomial(probs, num_samples=1)
            generated_token_logprob = torch.gather(logprobs, dim=1, index=generated_token)

            if callback is not None:
                callback([generated_token])
                
                
            context = torch.cat([context, generated_token], dim=1)
            if token_probs is None:
                token_probs = generated_token_logprob
            else:
                token_probs = torch.cat([token_probs, generated_token_logprob], dim=1)
                
            del logits, probs, logprobs, generated_token, generated_token_logprob   
            

        return context, token_probs
    
    # Get the log probability of a token given a context
    def get_token_logprob(self, context, token):
        logits, _ = self('forward', x=context[:, -self.params.block_size:])
        logits = logits[:,-1,:]
        logprobs = F.log_softmax(logits, dim=-1)
        token_logprob = torch.gather(logprobs, dim=1, index=token)  
        return token_logprob
    
    def disable_dropout(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train(False)

