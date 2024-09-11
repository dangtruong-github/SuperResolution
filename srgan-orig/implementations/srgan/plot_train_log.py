import matplotlib.pyplot as plt
import re
import pandas as pd

# Function to parse log lines and extract values
def parse_log_file(file_path):
    metrics = {
        'epoch': [],
        'train_generative_loss': [],
        'train_discriminative_loss': [],
        'val_generative_loss': [],
        'val_discriminative_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_lpips': [],
        'val_lr_psnr': []
    }
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regular expression to find epoch blocks
    pattern = re.compile(r'Epoch: (\d+)/\d+:\s*'
                         r'Train generative loss: ([\d.]+)\s*'
                         r'Train discriminative loss: ([\d.]+)\s*'
                         r'Validation generative loss: ([\d.]+)\s*'
                         r'Validation discriminative loss: ([\d.]+)\s*'
                         r'Validation PSNR score: ([\d.]+)\s*'
                         r'Validation SSIM score: ([\d.]+)\s*'
                         r'Validation LPIPS score: ([\d.]+)\s*'
                         r'Validation LR-PSNR score: ([\d.]+)', re.MULTILINE)
    
    for match in pattern.finditer(content):
        epoch = int(match.group(1))
        metrics['epoch'].append(epoch)
        metrics['train_generative_loss'].append(float(match.group(2)))
        metrics['train_discriminative_loss'].append(float(match.group(3)))
        metrics['val_generative_loss'].append(float(match.group(4)))
        metrics['val_discriminative_loss'].append(float(match.group(5)))
        metrics['val_psnr'].append(float(match.group(6)))
        metrics['val_ssim'].append(float(match.group(7)))
        metrics['val_lpips'].append(float(match.group(8)))
        metrics['val_lr_psnr'].append(float(match.group(9)))
    
    return pd.DataFrame(metrics)

# Path to your log file
file_path = 'train.log'

# Parse the log file
df = parse_log_file(file_path)

# Plotting the metrics
fig, axs = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Metrics over Epochs')

# Train Generative Loss
axs[0, 0].plot(df['epoch'], df['train_generative_loss'], label='Train Generative Loss', color='blue')
axs[0, 0].set_title('Train Generative Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# Train Discriminative Loss
axs[0, 1].plot(df['epoch'], df['val_lr_psnr'], label='Validation LR-PSNR Score', color='green')
axs[0, 1].set_title('Validation LR-PSNR Score')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('PSNR Score')
axs[0, 1].legend()

# Validation Generative Loss
axs[1, 0].plot(df['epoch'], df['val_generative_loss'], label='Validation Generative Loss', color='red')
axs[1, 0].set_title('Validation Generative Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Validation PSNR
axs[1, 1].plot(df['epoch'], df['val_psnr'], label='Validation PSNR', color='purple')
axs[1, 1].set_title('Validation PSNR')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('PSNR')
axs[1, 1].legend()

# Validation SSIM
axs[2, 0].plot(df['epoch'], df['val_ssim'], label='Validation SSIM', color='orange')
axs[2, 0].set_title('Validation SSIM')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('SSIM')
axs[2, 0].legend()

# Validation LPIPS
axs[2, 1].plot(df['epoch'], df['val_lpips'], label='Validation LPIPS', color='cyan')
axs[2, 1].set_title('Validation LPIPS')
axs[2, 1].set_xlabel('Epoch')
axs[2, 1].set_ylabel('LPIPS')
axs[2, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("train_log.png")
plt.show()
