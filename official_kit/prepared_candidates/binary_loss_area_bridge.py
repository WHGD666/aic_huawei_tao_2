def compute_binary_partition_loss(
    probabilities,
    expected_cut,
    net_cut_probabilities,
    net_weights,
    node_features,
    node_areas,
    balance_target,
    cut_weight,
    area_balance_weight,
    binarization_weight,
    epoch,
    max_epochs,
):
    """Establish soft area balance early, then use the proven final objective."""
    total_area = node_areas.sum()
    top_area_ratio = (node_areas * probabilities).sum() / total_area
    deviation = (top_area_ratio - balance_target).abs()

    # The stronger area shape helped one circuit but hurt the others when it
    # remained active throughout training. Use it only before binary hardening.
    base_area_loss = (deviation - 0.03).clamp(min=0.0).square()
    strict_area_loss = 4.0 * (deviation - 0.02).clamp(min=0.0).square()
    strict_fraction = min(1.0, max(0.0, 2.0 - 4.0 * epoch / max_epochs))
    area_loss = base_area_loss + strict_fraction * (strict_area_loss - base_area_loss)

    margin_error = (0.9 - (2.0 * probabilities - 1.0).abs()).clamp(min=0.0).square() / 0.81
    binary_loss = (
        0.25 * margin_error.mean()
        + 0.75 * (node_areas * margin_error).sum() / total_area
    )
    return (
        cut_weight * expected_cut
        + area_balance_weight * area_loss
        + binarization_weight * binary_loss
    )
