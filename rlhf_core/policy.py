import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, Tuple, Optional
import numpy as np


class PolicyModel(nn.Module):
    """A thin wrapper around GPT-2 for RLHF policy training."""
    
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(device)
        self.model.eval()  # Start in eval mode
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits and other outputs."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }
    
    def get_logprobs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get log probabilities for the input sequence."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            logprobs = F.log_softmax(logits, dim=-1)
            
        # Get logprobs for the tokens that follow each position
        shifted_input_ids = input_ids[:, 1:]
        shifted_logprobs = logprobs[:, :-1]
        
        # Gather logprobs for the actual next tokens
        gathered_logprobs = torch.gather(shifted_logprobs, -1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        
        return gathered_logprobs
    
    def sample(self, 
               prompt_ids: torch.Tensor, 
               max_new_tokens: int = 20,
               temperature: float = 1.0,
               top_p: float = 0.9,
               do_sample: bool = True,
               generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample new tokens given a prompt."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate new tokens with deterministic sampling
            generate_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.eos_token_id,
                'return_dict_in_generate': True,
                'output_scores': True
            }
            
            # For determinism, set a fixed seed if generator is provided
            if generator is not None:
                # Set the seed for the model's internal random state
                torch.manual_seed(generator.initial_seed())
            
            generated = self.model.generate(
                prompt_ids,
                **generate_kwargs
            )
            
            # Get the full sequence and scores
            full_sequence = generated.sequences
            scores = generated.scores
            
            # Calculate logprobs for the new tokens
            new_token_logprobs = []
            for i, score in enumerate(scores):
                logprobs = F.log_softmax(score, dim=-1)
                next_token_id = full_sequence[:, prompt_ids.shape[1] + i]
                token_logprobs = torch.gather(logprobs, -1, next_token_id.unsqueeze(-1)).squeeze(-1)
                new_token_logprobs.append(token_logprobs)
            
            new_token_logprobs = torch.stack(new_token_logprobs, dim=1)
            
        return full_sequence, new_token_logprobs
    
    def get_kl_divergence(self, 
                          input_ids: torch.Tensor, 
                          reference_model: 'PolicyModel',
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute KL divergence between current policy and reference model."""
        current_logprobs = self.get_logprobs(input_ids, attention_mask)
        reference_logprobs = reference_model.get_logprobs(input_ids, attention_mask)
        
        # KL divergence: KL(p||q) = E_p[log p - log q]
        kl_div = F.kl_div(
            F.log_softmax(current_logprobs, dim=-1),
            F.softmax(reference_logprobs, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        
        return kl_div
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def get_trainable_params(self):
        """Get trainable parameters for optimization."""
        return self.model.parameters()
    
    def set_train_mode(self, train: bool = True):
        """Set training mode."""
        self.model.train(train)