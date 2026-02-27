import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class TransformerModel(nn.Module):
    """
    Decision Transformer for portfolio optimization using a quantized LLM backbone.
    """
    def __init__(self, model_id: str, nb_tickers: int, logits_clamp_value: float) -> None:
        """
        Args:
            model_id: model_id: HuggingFace hub ID for the base transformer model.
            nb_tickers: Number of tradable assets in the dataset.
            logits_clamp_value: Max/Min threshold for raw logits.
        """
        super().__init__()

        self.logits_clamp_value = logits_clamp_value
        self.dtype = torch.bfloat16

        # Configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Dimensionality constraints
        output_dim_0 = self.model.config.hidden_size
        output_dim_1 = nb_tickers + 1  # Number of stocks + 1 cash weight

        # Configuration for Low Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None,
        )

        # Apply LoRA adapters to the quantized backbone
        self.model = get_peft_model(self.model, lora_config)

        # Bridge
        self.state_encoder = nn.Linear(nb_tickers, output_dim_0).to(self.dtype)

        # Head
        self.action_logits = nn.Linear(output_dim_0, output_dim_1).to(self.dtype)

        # Weight Initialization
        torch.nn.init.xavier_uniform_(self.action_logits.weight, gain=0.01)
        torch.nn.init.zeros_(self.action_logits.bias)


    def forward(self, x: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict portfolio weights from financial states.

        Args:
            x: Time series (financial features), of shape (batch_size, context_length, nb_tickers).
            temperature: Scaling factor for the softmax distribution.
                Higher values lead to more uniform weights; lower values lead to higher sparsity.

        Returns:
            Portfolio weights, expressed as softmax probabilities, of shape (batch_size, context_length, nb_tickers +1).
            Pre-softmax scores after clamping of shape (batch_size, context_length, nb_tickers +1).
        """
        # Ensure input precision matches model parameters (e.g., bfloat16)
        x = x.to(self.dtype)

        # Map raw financial features into the LLM's high-dimensional embedding space
        embeddings = self.state_encoder(x)

        # Pass embeddings through the transformer backbone
        outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)

        # Extract final layer hidden states representing the temporal context
        latent_context = outputs.hidden_states[-1]

        # Project latent features to action space (pre-softmax scores)
        logits = self.action_logits(latent_context)

        # Clip the logits to prevent any single weight to approach 1.0, crushing other weights
        logits_clamped = torch.clamp(logits, min=-self.logits_clamp_value, max=self.logits_clamp_value)

        # Temperature scaling to adjusts the sharpness of the distribution
        logits_scaled = logits_clamped / temperature

        # Apply softmax to compute probabilities
        portfolio_weights = torch.softmax(logits_scaled, dim=-1)

        return portfolio_weights, logits_clamped
