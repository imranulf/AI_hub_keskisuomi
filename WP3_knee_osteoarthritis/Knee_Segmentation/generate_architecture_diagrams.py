import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# CONFIGURATION
# ==========================================
output_dir = 'architecture_diagrams'
os.makedirs(output_dir, exist_ok=True)

# Color Palette inspired by medical imaging & deep learning paper architectures
colors = {
    'input': '#E8EAED',        
    'conv': '#42A5F5',         # Blue for feature extractors
    'pool': '#FFA726',         # Orange for pooling/reduction
    'block': '#66BB6A',        # Green for residual/mbconv/swin blocks
    'attention': '#AB47BC',    # Purple for attention/merging
    'fc': '#EF5350',           # Red for dense layers
    'output': '#7E57C2'        # Deep purple for output
}

# ==========================================
# ADVANCED CNN-STYLE DIAGRAM GENERATOR
# ==========================================
def draw_paper_style_architecture(model_name, feature_maps, output_filename):
    """
    Draws a left-to-right feature map architecture diagram.
    Simulates spatial dimension reductions (height) and channel expansions (width).
    """
    fig, ax = plt.subplots(figsize=(24, 6))  # Widened figure out from 18 to 24 to prevent overlap
    ax.axis('off')
    
    # Calculate total width to set x_lim
    total_x = sum([fm['w'] + fm.get('spacing', 1.0) for fm in feature_maps]) + 2
    ax.set_xlim(0, total_x)
    max_h = max([fm['h'] for fm in feature_maps])
    ax.set_ylim(-max_h/2 - 2, max_h/2 + 3)
    
    plt.title(f"Proposed {model_name} Network Architecture for Knee Osteoarthritis Detection", 
              fontsize=20, fontweight='bold', pad=20)
              
    current_x = 1.0
    
    for i, fm in enumerate(feature_maps):
        h = fm['h']
        w = fm['w']
        c = colors.get(fm['type'], '#CCCCCC')
        
        # Override rotation if specified
        rot = fm.get('rot', 90 if h > 2 else 0)
        
        # Draw the feature map 'volume' as a colored Box
        # We center it vertically around Y=0
        y_bottom = -h / 2.0
        
        # Shadow/3D effect offset
        offset = 0.15
        if fm['type'] != 'input' and fm['type'] != 'output':
            # Back shadow rectangle
            shadow = patches.Rectangle((current_x + offset, y_bottom - offset), w, h, 
                                       linewidth=1, edgecolor='#555555', facecolor='#DDDDDD')
            ax.add_patch(shadow)
            
        # Main rectangle
        rect = patches.Rectangle((current_x, y_bottom), w, h, 
                                 linewidth=1.5, edgecolor='black', facecolor=c)
        ax.add_patch(rect)
        
        # Top Label (e.g. "Conv1", "ResBlock")
        plt.text(current_x + w/2, h/2 + 0.3, fm['name'], 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', rotation=0)
                 
        # Inside Label (e.g. Tensor dims)
        plt.text(current_x + w/2, 0, fm['desc'], 
                 ha='center', va='center', fontsize=9, color='white' if fm['type'] in ['conv', 'fc', 'attention', 'block'] else 'black', 
                 fontweight='bold', rotation=rot)
        
        # Bottom Label (Channels / Modifications)
        if 'bottom_label' in fm:
            plt.text(current_x + w/2, -h/2 - 0.4, fm['bottom_label'], 
                     ha='center', va='top', fontsize=10, color='#d32f2f' if 'Modified' in fm['bottom_label'] else 'black',
                     fontweight='bold' if 'Modified' in fm['bottom_label'] else 'normal')
                     
        # Draw connecting arrow to the next block
        if i < len(feature_maps) - 1:
            next_spacing = fm.get('spacing', 1.0)
            start_x = current_x + w
            end_x = start_x + next_spacing - 0.1
            
            # Simple arrow
            ax.annotate('', xy=(end_x, 0), xytext=(start_x, 0),
                        arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8))
                        
            if 'arrow_label' in fm:
                plt.text(start_x + next_spacing/2, 0.2, fm['arrow_label'], 
                         ha='center', va='bottom', fontsize=9, fontstyle='italic')
                         
        current_x += w + fm.get('spacing', 1.0)
        
    # Legend
    legend_elements = [
        patches.Patch(facecolor=colors['input'], edgecolor='black', label='Input/Output Data'),
        patches.Patch(facecolor=colors['conv'], edgecolor='black', label='Convolutional Layer'),
        patches.Patch(facecolor=colors['pool'], edgecolor='black', label='Pooling (Max/Avg)'),
        patches.Patch(facecolor=colors['block'], edgecolor='black', label='Core Blocks (ResNet/MBConv/Swin)'),
        patches.Patch(facecolor=colors['fc'], edgecolor='black', label='Fully Connected / Dense Head')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
              ncol=5, fontsize=12, frameon=False)
              
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated paper-style diagram: {output_filename}")


# ==========================================
# DEFINE MODEL STRUCTURES (Spatial representations)
# ==========================================

# Fixed block widths and rotations to guarantee NO overflow text
resnet_paper = [
    {'name': 'Input Image', 'desc': 'Grayscale X-Ray\n224x224', 'bottom_label': '1 Channel', 'type': 'input', 'h': 6, 'w': 1.6, 'spacing': 1.5, 'rot': 0},
    {'name': 'Conv1', 'desc': '7x7 Conv\nStride 2', 'bottom_label': '*Modified: 1-Ch*', 'type': 'conv', 'h': 5.5, 'w': 1.2, 'spacing': 1.2, 'rot': 90},
    {'name': 'MaxPool', 'desc': '3x3, /2', 'bottom_label': '64', 'type': 'pool', 'h': 4.5, 'w': 1.0, 'spacing': 1.2, 'rot': 90},
    {'name': 'Stage 1\n(Conv2_x)', 'desc': 'BasicBlock x2', 'bottom_label': '64', 'type': 'block', 'h': 4.5, 'w': 1.5, 'spacing': 1.5, 'rot': 90},
    {'name': 'Stage 2\n(Conv3_x)', 'desc': 'BasicBlock x2', 'bottom_label': '128', 'type': 'block', 'h': 3.5, 'w': 1.5, 'spacing': 1.5, 'rot': 90},
    {'name': 'Stage 3\n(Conv4_x)', 'desc': 'BasicBlock x2', 'bottom_label': '256', 'type': 'block', 'h': 2.8, 'w': 1.8, 'spacing': 1.5, 'rot': 0},
    {'name': 'Stage 4\n(Conv5_x)', 'desc': 'BasicBlock x2', 'bottom_label': '512', 'type': 'block', 'h': 2.0, 'w': 2.2, 'spacing': 1.5, 'rot': 0},
    {'name': 'GAP', 'desc': 'Avg Pool', 'bottom_label': '512 Vector', 'type': 'pool', 'h': 1.5, 'w': 1.2, 'spacing': 1.5, 'rot': 0},
    {'name': 'FC Layer', 'desc': 'Linear', 'bottom_label': '*Replaced Head*', 'type': 'fc', 'h': 4.0, 'w': 1.2, 'spacing': 1.5, 'rot': 90},
    {'name': 'Output', 'desc': 'Softmax\nKL0 / KL2', 'bottom_label': '2 Classes', 'type': 'output', 'h': 1.2, 'w': 1.5, 'rot': 0}
]

efficientnet_paper = [
    {'name': 'Input Image', 'desc': 'Grayscale X-Ray\n224x224', 'bottom_label': '1 Channel', 'type': 'input', 'h': 6, 'w': 1.6, 'spacing': 1.5, 'rot': 0},
    {'name': 'Stem Matrix', 'desc': 'Conv2dNormAct', 'bottom_label': '*Modified: 1-Ch*', 'type': 'conv', 'h': 5.5, 'w': 1.4, 'spacing': 1.2, 'rot': 90},
    {'name': 'Stage 1-3\n(MBConv)', 'desc': 'Squeeze/Excite', 'bottom_label': '16->40', 'type': 'block', 'h': 4.5, 'w': 1.6, 'spacing': 1.2, 'rot': 90},
    {'name': 'Stage 4-5\n(MBConv)', 'desc': 'Expansion 6', 'bottom_label': '80->112', 'type': 'block', 'h': 3.5, 'w': 1.6, 'spacing': 1.2, 'rot': 90},
    {'name': 'Stage 6-7\n(MBConv)', 'desc': 'Deep Features', 'bottom_label': '192->320', 'type': 'block', 'h': 2.5, 'w': 1.8, 'spacing': 1.5, 'rot': 0},
    {'name': 'Top CNN', 'desc': 'Conv2d + Pool', 'bottom_label': '1280 Vector', 'type': 'pool', 'h': 1.8, 'w': 1.6, 'spacing': 1.5, 'rot': 0},
    {'name': 'Dense Drop', 'desc': 'Linear + Drop', 'bottom_label': '*Replaced Head*', 'type': 'fc', 'h': 4.0, 'w': 1.4, 'spacing': 1.5, 'rot': 90},
    {'name': 'Output', 'desc': 'Softmax\nKL0 / KL2', 'bottom_label': '2 Classes', 'type': 'output', 'h': 1.2, 'w': 1.5, 'rot': 0}
]

swin_paper = [
    {'name': 'Input Image', 'desc': 'Grayscale X-Ray\n224x224', 'bottom_label': '1 Channel', 'type': 'input', 'h': 6, 'w': 1.6, 'spacing': 1.5, 'rot': 0},
    {'name': 'Patch Embed', 'desc': 'Conv2d 4x4', 'bottom_label': '*Modified: 1-Ch*', 'type': 'attention', 'h': 5.5, 'w': 1.4, 'spacing': 1.5, 'rot': 90},
    {'name': 'Stage 1', 'desc': 'Swin Blk x2', 'bottom_label': 'W-MSA (96)', 'type': 'block', 'h': 4.8, 'w': 1.6, 'spacing': 1.2, 'rot': 90},
    {'name': 'Patch Merge', 'desc': 'Downsample', 'bottom_label': '192', 'type': 'pool', 'h': 4.0, 'w': 1.2, 'spacing': 1.0, 'rot': 90},
    {'name': 'Stage 2', 'desc': 'Swin Blk x2', 'bottom_label': 'SW-MSA (192)', 'type': 'block', 'h': 3.5, 'w': 1.6, 'spacing': 1.2, 'rot': 90},
    {'name': 'Patch Merge', 'desc': 'Downsample', 'bottom_label': '384', 'type': 'pool', 'h': 2.5, 'w': 1.2, 'spacing': 1.0, 'rot': 90},
    {'name': 'Stage 3', 'desc': 'Swin Blk x6', 'bottom_label': 'SW-MSA (384)', 'type': 'block', 'h': 2.2, 'w': 1.8, 'spacing': 1.2, 'rot': 0},
    {'name': 'Stage 4 / Pool', 'desc': 'Swin Blk x2', 'bottom_label': 'Linear (768)', 'type': 'pool', 'h': 1.6, 'w': 1.5, 'spacing': 1.5, 'rot': 0},
    {'name': 'Head Matrix', 'desc': 'Linear Layer', 'bottom_label': '*Replaced Head*', 'type': 'fc', 'h': 4.0, 'w': 1.2, 'spacing': 1.5, 'rot': 90},
    {'name': 'Output', 'desc': 'Softmax\nKL0 / KL2', 'bottom_label': '2 Classes', 'type': 'output', 'h': 1.2, 'w': 1.5, 'rot': 0}
]

if __name__ == '__main__':
    print("Generating Academic Paper Volume Diagrams...")
    draw_paper_style_architecture("ResNet-18", resnet_paper, "ResNet18_Paper_Architecture.png")
    draw_paper_style_architecture("EfficientNet-B0", efficientnet_paper, "EfficientNetB0_Paper_Architecture.png")
    draw_paper_style_architecture("Swin-Tiny", swin_paper, "SwinTiny_Paper_Architecture.png")
    print(f"Successfully saved volumetric diagrams to '{output_dir}/'.")
