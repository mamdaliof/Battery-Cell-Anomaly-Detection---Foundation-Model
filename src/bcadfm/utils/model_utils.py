import logging
from torch import nn

logger = logging.getLogger(__name__)

def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Count total, trainable, and non-trainable parameters in a model.

    Returns a dictionary with keys:
        - "total": total number of parameters
        - "trainable": number of trainable parameters
        - "non_trainable": number of non-trainable parameters
        - "percentage_trainable": percentage of parameters that are trainable
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    percentage = (trainable / total) * 100 if total > 0 else 0.0

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "percentage_trainable": percentage,
    }

def log_parameter_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Log and print a detailed summary of the model's parameters."""
    summary = count_parameters(model)
    logger.info(f"=== {model_name} Parameter Summary ===")
    logger.info(f"Total Parameters:          {summary['total']:,}")
    logger.info(f"Trainable Parameters:      {summary['trainable']:,}")
    logger.info(f"Non-Trainable Parameters:  {summary['non_trainable']:,}")
    logger.info(f"Trainable %:              {summary['percentage_trainable']:.4f}%")
    logger.info("=====================================")

    # Print to console for direct training run visibility
    print(f"\n=== {model_name} Parameter Summary ===")
    print(f"Total Parameters:          {summary['total']:,}")
    print(f"Trainable Parameters:      {summary['trainable']:,}")
    print(f"Non-Trainable Parameters:  {summary['non_trainable']:,}")
    print(f"Trainable %:              {summary['percentage_trainable']:.4f}%")
    print("=====================================\n")
