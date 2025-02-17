import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer
)

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

# Top-level model matching the Hugging Face naming.
class LlamaForCausalLMEagle(PreTrainedModel):
    def __init__(self, config):
        """
        Initializes the model with the Hugging Face structure:
        
          LlamaForCausalLM(
            (model): LlamaModel(...)
            (lm_head): Linear(hidden_size, vocab_size, bias=False)
            (draft_fc): Linear(2*hidden_size, hidden_size, bias=False)  # for speculative decoding
          )
        """
        super().__init__(config)
        self.gradient_checkpointing = True

        config.attn_implementation="flash_attention_2"
        self.model = LlamaModel(config)
        # print(self.model.dtype)
        
        # don't actually need to project to token space
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # This projection layer maps the concatenated [token_emb; full_hidden] (of size 2*hidden_size)
        # down to hidden_size.
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False, dtype=config.torch_dtype)
    
    def load_embedding_weights(self, weights):
        self.model.embed_tokens.weight = nn.Parameter(weights)

    def forward(
            self,
            hidden_state: torch.Tensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_emb = self.model.embed_tokens(input_ids)
        concat = torch.cat([token_emb, hidden_state], dim=-1).to(torch.bfloat16)
        proj = self.fc(concat)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            inputs_embeds=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        return hidden_states

    # def forward(self, input_ids, attention_mask=None):
    #     # Standard forward pass (e.g. for full-causal LM generation)
    #     hidden_states = self.model(input_ids, attention_mask=attention_mask)
    #     logits = self.lm_head(hidden_states)
    #     return logits

    # def forward_draft(self, token_ids, full_hidden):
    #     """
    #     Forward pass for the Eagle Speculative Decoding draft branch.
        
    #     Args:
    #         token_ids (torch.LongTensor): Input token IDs of shape (batch, 1).
    #         full_hidden (torch.Tensor): Full model's final hidden state for the current token,
    #                                     shape (batch, hidden_size) or (batch, 1, hidden_size).
        
    #     Returns:
    #         torch.Tensor: Logits over the vocabulary with shape (batch, vocab_size).
    #     """
    #     # Obtain the token embedding from the shared embed_tokens.
    #     token_emb = self.model.embed_tokens(token_ids)  # (batch, 1, hidden_size)
        
    #     # Ensure full_hidden has a sequence dimension.
    #     if full_hidden.dim() == 2:
    #         full_hidden = full_hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        
    #     # Concatenate token embedding and full_hidden along the hidden dimension.
    #     concat = torch.cat([token_emb, full_hidden], dim=-1)  # (batch, 1, 2*hidden_size)
        
    #     # Project the concatenated vector to hidden_size.
    #     proj = self.draft_fc(concat)
        
    #     # Process the result with the single transformer (decoder) layer.
    #     out = self.model.layers[0](proj)
    #     out = self.model.norm(out)
        
    #     # Project to vocabulary logits using the shared lm_head weight.
    #     logits = F.linear(out, self.lm_head.weight)
        
    #     # Remove the sequence dimension (assumed to be 1).
    #     return logits.squeeze(1)