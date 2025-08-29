import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional
import re


class ToyRewardModel(nn.Module):
    """A lightweight toy reward model for RLHF training."""
    
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(device)
        self.model.eval()
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning reward scores."""
        # Get the last hidden state
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Use the last token's hidden state for classification
        last_token_hidden = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
        
        # Simple linear projection to scalar reward
        reward_head = nn.Linear(last_token_hidden.size(-1), 1).to(self.device)
        rewards = reward_head(last_token_hidden).squeeze(-1)  # [batch_size]
        
        return rewards
    
    def compute_reward(self, 
                       input_ids: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None,
                       reward_type: str = "sentiment") -> torch.Tensor:
        """Compute rewards based on different criteria."""
        if reward_type == "sentiment":
            return self._sentiment_reward(input_ids, attention_mask)
        elif reward_type == "preference":
            return self._preference_reward(input_ids, attention_mask)
        elif reward_type == "length":
            return self._length_reward(input_ids, attention_mask)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    def _sentiment_reward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute sentiment-based rewards."""
        # Decode sequences to text
        sequences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        rewards = []
        for seq in sequences:
            # Simple keyword-based sentiment scoring
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'disappointing']
            
            seq_lower = seq.lower()
            positive_count = sum(1 for word in positive_words if word in seq_lower)
            negative_count = sum(1 for word in negative_words if word in seq_lower)
            
            # Reward based on sentiment balance
            if positive_count > negative_count:
                reward = 2.0 + (positive_count - negative_count) * 0.5
            elif negative_count > positive_count:
                reward = -1.0 - (negative_count - positive_count) * 0.5
            else:
                reward = 0.0
                
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def _preference_reward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute preference-based rewards (higher for specific endings)."""
        sequences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        rewards = []
        for seq in sequences:
            # Reward sequences that end with positive phrases
            positive_endings = ['thank you', 'great', 'good', 'excellent', 'wonderful']
            seq_lower = seq.lower().strip()
            
            reward = 0.0
            for ending in positive_endings:
                if seq_lower.endswith(ending):
                    reward = 3.0
                    break
            
            # Small penalty for very short sequences
            if len(seq.split()) < 3:
                reward -= 1.0
                
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def _length_reward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute length-based rewards (encourage moderate length)."""
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1)
        else:
            seq_lengths = torch.ones(input_ids.size(0), dtype=torch.long, device=self.device) * input_ids.size(1)
        
        # Reward moderate lengths (5-15 tokens), penalize very short or very long
        rewards = torch.where(
            seq_lengths < 5,
            -2.0,  # Too short
            torch.where(
                seq_lengths > 20,
                -1.0,  # Too long
                1.0    # Good length
            )
        )
        
        return rewards.float()
    
    def get_reward_stats(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Get statistics about the rewards."""
        return {
            'mean': rewards.mean().item(),
            'std': rewards.std().item(),
            'min': rewards.min().item(),
            'max': rewards.max().item(),
            'median': rewards.median().item()
        }


class SimpleClassifierReward(nn.Module):
    """An even simpler classifier-based reward model."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 64, device: str = "cpu"):
        super().__init__()
        self.device = device
        
        # Simple feedforward network
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.to(device)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reward scores."""
        # Simple pooling: average embeddings
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        pooled = embeddings.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classify to get reward
        rewards = self.classifier(pooled).squeeze(-1)  # [batch_size]
        
        return rewards
    
    def compute_reward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute rewards using the classifier."""
        return self.forward(input_ids)