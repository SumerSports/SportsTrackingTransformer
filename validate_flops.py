"""
Manual FLOP Validation Script (Scratch - Not for Commit)

This script manually calculates expected FLOPs for transformer attention
to validate that calflops is counting correctly.

Based on formulas from:
- https://www.adamcasson.com/posts/transformer-flops
- https://github.com/google-research/electra/blob/master/flops_computation.py
"""

import sys
sys.path.insert(0, 'src')

import torch
from calflops import calculate_flops
from models import LitModel


def manual_attention_flops(seq_len: int, model_dim: int, num_heads: int) -> int:
    """
    Manually calculate FLOPs for a single transformer attention layer.

    Formula breakdown:
    - QKV projection: 2 * seq_len * 3 * model_dim * model_dim
    - Attention logits (Q @ K^T): 2 * seq_len * seq_len * model_dim
    - Attention reduction ((QK) @ V): 2 * seq_len * seq_len * model_dim
    - Output projection: 2 * seq_len * model_dim * model_dim

    Note: The factor of 2 comes from each matmul being a multiply-accumulate
    operation (one multiply + one add = 2 FLOPs per element).
    """
    qkv_proj = 2 * seq_len * 3 * model_dim * model_dim
    attn_logits = 2 * seq_len * seq_len * model_dim
    attn_reduce = 2 * seq_len * seq_len * model_dim
    output_proj = 2 * seq_len * model_dim * model_dim

    total = qkv_proj + attn_logits + attn_reduce + output_proj

    print(f"Manual Calculation for Single Attention Layer:")
    print(f"  seq_len={seq_len}, model_dim={model_dim}, num_heads={num_heads}")
    print(f"  QKV projection:     {qkv_proj:>15,} FLOPs")
    print(f"  Attention logits:   {attn_logits:>15,} FLOPs")
    print(f"  Attention reduction:{attn_reduce:>15,} FLOPs")
    print(f"  Output projection:  {output_proj:>15,} FLOPs")
    print(f"  Total per layer:    {total:>15,} FLOPs")
    print()

    return total


def manual_feedforward_flops(seq_len: int, model_dim: int, dim_feedforward: int) -> int:
    """
    Manually calculate FLOPs for feedforward layers in transformer.

    Two linear layers: model_dim -> dim_feedforward -> model_dim
    """
    linear1 = 2 * seq_len * model_dim * dim_feedforward
    linear2 = 2 * seq_len * dim_feedforward * model_dim

    total = linear1 + linear2

    print(f"Manual Calculation for Feedforward Layer:")
    print(f"  Linear1 (expand):   {linear1:>15,} FLOPs")
    print(f"  Linear2 (contract): {linear2:>15,} FLOPs")
    print(f"  Total per layer:    {total:>15,} FLOPs")
    print()

    return total


def manual_zoo_flops(num_offense: int, num_defense: int, feature_len: int,
                     model_dim: int, num_layers: int) -> int:
    """
    Manually calculate FLOPs for TheZooArchitecture model.

    Architecture breakdown:
    1. Feature embedding: feature_len -> model_dim for each O*D interaction
    2. FF Block 1: num_layers of Conv2d (1x1 convolutions across O*D interactions)
    3. Pooling across offense dimension (no FLOPs)
    4. FF Block 2: num_layers of Conv1d (1x1 convolutions across D dimension)
    5. Pooling across defense dimension (no FLOPs)
    6. Output decoder: series of linear layers
    """
    num_interactions = num_offense * num_defense

    # 1. Feature embedding: Linear(feature_len, model_dim) for each interaction
    embedding_flops = 2 * num_interactions * feature_len * model_dim

    # 2. FF Block 1: num_layers of Conv2d(model_dim, model_dim, kernel=(1,1))
    # Each Conv2d with 1x1 kernel is equivalent to a linear layer per spatial location
    ff_block1_flops = num_layers * (2 * num_interactions * model_dim * model_dim)

    # 3. After pooling across offense dimension, we have (model_dim, defense) shape
    # FF Block 2: num_layers of Conv1d(model_dim, model_dim, kernel=1)
    ff_block2_flops = num_layers * (2 * num_defense * model_dim * model_dim)

    # 4. Output decoder layers (after pooling to model_dim vector)
    # Based on TheZooArchitecture code:
    # - (num_layers - 2) layers of Linear(model_dim, model_dim) if num_layers >= 2
    # - Linear(model_dim, model_dim // 4)
    # - Linear(model_dim // 4, 2)
    num_decoder_layers = max(0, num_layers - 2)
    decoder_hidden_flops = num_decoder_layers * (2 * model_dim * model_dim)
    decoder_compress_flops = 2 * model_dim * (model_dim // 4)
    decoder_output_flops = 2 * (model_dim // 4) * 2
    decoder_flops = decoder_hidden_flops + decoder_compress_flops + decoder_output_flops

    total = embedding_flops + ff_block1_flops + ff_block2_flops + decoder_flops

    print(f"Manual Calculation for Zoo Model:")
    print(f"  num_offense={num_offense}, num_defense={num_defense}, feature_len={feature_len}")
    print(f"  model_dim={model_dim}, num_layers={num_layers}")
    print(f"  num_interactions={num_interactions}")
    print()
    print(f"  Feature embedding:  {embedding_flops:>15,} FLOPs")
    print(f"  FF Block 1 (Conv2d):{ff_block1_flops:>15,} FLOPs")
    print(f"  FF Block 2 (Conv1d):{ff_block2_flops:>15,} FLOPs")
    print(f"  Decoder layers:     {decoder_flops:>15,} FLOPs")
    print(f"  Total:              {total:>15,} FLOPs")
    print()

    return total


def validate_transformer_model():
    """Validate transformer FLOP counts."""
    print("="*80)
    print("TRANSFORMER MODEL VALIDATION")
    print("="*80)
    print()

    # Load model
    checkpoint_path = 'models/best_models/transformer/best_model.ckpt'
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = lit_model.model
    model.eval()

    # Get hyperparameters
    seq_len = 22
    model_dim = lit_model.hparams['model_dim']
    num_layers = lit_model.hparams['num_layers']
    num_heads = lit_model.hparams['num_heads']
    dim_feedforward = lit_model.hparams['dim_feedforward']

    print(f"Model Configuration:")
    print(f"  seq_len: {seq_len}")
    print(f"  model_dim: {model_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  dim_feedforward: {dim_feedforward}")
    print()

    # Manual calculation for transformer encoder
    print("-"*80)
    print("MANUAL FLOP CALCULATIONS")
    print("-"*80)
    print()

    attn_flops_per_layer = manual_attention_flops(seq_len, model_dim, num_heads)
    ff_flops_per_layer = manual_feedforward_flops(seq_len, model_dim, dim_feedforward)
    encoder_flops_per_layer = attn_flops_per_layer + ff_flops_per_layer
    total_encoder_flops = encoder_flops_per_layer * num_layers

    print(f"Per Encoder Layer:  {encoder_flops_per_layer:>15,} FLOPs")
    print(f"Total Encoder ({num_layers} layers): {total_encoder_flops:>15,} FLOPs")
    print()

    # Get calflops measurement
    print("-"*80)
    print("CALFLOPS MEASUREMENT")
    print("-"*80)
    print()

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, seq_len, 6),
        print_results=False,
        output_as_string=False
    )

    print(f"Total Model FLOPs:  {flops:>15,} FLOPs")
    print(f"Total Model MACs:   {macs:>15,} MACs")
    print()

    # Compare
    print("-"*80)
    print("VALIDATION")
    print("-"*80)
    print()

    # The total FLOPs should be higher than just encoder (due to embedding + decoder)
    print(f"Manual encoder estimate:    {total_encoder_flops:>15,} FLOPs")
    print(f"calflops full model count:  {flops:>15,} FLOPs")
    print(f"Difference (embedding+decoder): {flops - total_encoder_flops:>15,} FLOPs")
    print()

    if total_encoder_flops < flops:
        print("✓ PASS: calflops count is higher than manual encoder estimate")
        print("  (This is expected - calflops includes embedding + decoder layers)")
    else:
        print("✗ FAIL: calflops count is lower than manual encoder estimate")
        print("  (This suggests calflops may not be counting attention properly)")
    print()

    # Check if attention FLOPs are a significant portion
    attention_proportion = (attn_flops_per_layer * num_layers) / flops
    print(f"Attention FLOPs as % of total: {attention_proportion*100:.1f}%")
    if attention_proportion > 0.1:
        print("✓ PASS: Attention operations are a significant portion (>10%)")
    else:
        print("✗ FAIL: Attention operations seem underrepresented (<10%)")
    print()


def validate_zoo_model():
    """Validate zoo model FLOP counts."""
    print("="*80)
    print("ZOO MODEL VALIDATION")
    print("="*80)
    print()

    # Load model
    checkpoint_path = 'models/best_models/zoo/best_model.ckpt'
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = lit_model.model
    model.eval()

    # Get hyperparameters
    num_offense = 10
    num_defense = 11
    feature_len = 10
    model_dim = lit_model.hparams['model_dim']
    num_layers = lit_model.hparams['num_layers']

    print(f"Model Configuration:")
    print(f"  num_offense: {num_offense}")
    print(f"  num_defense: {num_defense}")
    print(f"  feature_len: {feature_len}")
    print(f"  model_dim: {model_dim}")
    print(f"  num_layers: {num_layers}")
    print()

    # Manual calculation
    print("-"*80)
    print("MANUAL FLOP CALCULATIONS")
    print("-"*80)
    print()

    manual_flops = manual_zoo_flops(num_offense, num_defense, feature_len,
                                     model_dim, num_layers)

    # Get calflops measurement
    print("-"*80)
    print("CALFLOPS MEASUREMENT")
    print("-"*80)
    print()

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, num_offense, num_defense, feature_len),
        print_results=False,
        output_as_string=False
    )

    print(f"Total Model FLOPs:  {flops:>15,} FLOPs")
    print(f"Total Model MACs:   {macs:>15,} MACs")
    print()

    # Compare
    print("-"*80)
    print("VALIDATION")
    print("-"*80)
    print()

    print(f"Manual estimate:            {manual_flops:>15,} FLOPs")
    print(f"calflops count:             {flops:>15,} FLOPs")
    difference = flops - manual_flops
    print(f"Difference (norm+other):    {difference:>15,} FLOPs")
    print(f"Difference percentage:      {abs(difference)/flops*100:>15.1f}%")
    print()

    # Manual calculation should be close to calflops (within 20% for norm layers, etc.)
    if abs(difference) / flops < 0.20:
        print("✓ PASS: calflops count is within 20% of manual estimate")
        print("  (Difference likely due to BatchNorm/LayerNorm operations)")
    else:
        print("✗ FAIL: calflops count differs significantly from manual estimate")
        print("  (Suggests calflops may not be counting operations correctly)")
    print()


def compare_models():
    """Compare FLOP counts between zoo and transformer."""
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print()

    # Zoo model
    print("Zoo Model:")
    zoo_model = LitModel.load_from_checkpoint('models/best_models/zoo/best_model.ckpt', map_location='cpu')
    zoo_flops, _, _ = calculate_flops(
        model=zoo_model.model,
        input_shape=(1, 10, 11, 10),
        print_results=False,
        output_as_string=False
    )
    print(f"  FLOPs: {zoo_flops:>15,}")
    print()

    # Transformer model
    print("Transformer Model:")
    transformer_model = LitModel.load_from_checkpoint('models/best_models/transformer/best_model.ckpt', map_location='cpu')
    transformer_flops, _, _ = calculate_flops(
        model=transformer_model.model,
        input_shape=(1, 22, 6),
        print_results=False,
        output_as_string=False
    )
    print(f"  FLOPs: {transformer_flops:>15,}")
    print()

    # Ratio
    ratio = transformer_flops / zoo_flops
    print(f"Ratio (transformer/zoo): {ratio:.2f}x")
    print()


if __name__ == "__main__":
    validate_transformer_model()
    validate_zoo_model()
    compare_models()
