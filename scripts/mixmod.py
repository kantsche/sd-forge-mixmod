import gradio as gr
import torch
import os
import traceback
import gc
from backend.sampling.condition import Condition, compile_conditions, compile_weighted_conditions
from modules import prompt_parser, sd_samplers_common

from modules import scripts, shared, sd_models, devices
from backend.patcher.base import set_model_options_post_cfg_function
from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend.loader import forge_loader
from backend.utils import load_torch_file
from backend import memory_management

# Global secondary model - keeping a reference prevents garbage collection
secondary_model = None

def is_model_on_cpu():
    """Check if the secondary model is currently on CPU"""
    global secondary_model
    
    if secondary_model is None:
        return False
        
    try:
        # More thorough check for device status
        for name, module in secondary_model.forge_objects.unet.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.device.type == 'cpu':
                    return True
        return False
    except Exception as e:
        print(f"MixMod: Error checking model device: {e}")
        return True  # Assume CPU to be safe

def ensure_model_on_gpu():
    """Ensure the model is fully on GPU"""
    global secondary_model
    
    if secondary_model is None:
        return False
        
    try:
        # Explicitly move the entire model to GPU
        secondary_model.forge_objects.unet.model.to(devices.device)
        print(f"MixMod: Model explicitly moved to {devices.device}")
        return True
    except Exception as e:
        print(f"MixMod: Error moving model to GPU: {e}")
        return False

def load_secondary_model(model_path, force_reload=False):
    global secondary_model
    
    # If force_reload is set, unload the model first
    if force_reload and secondary_model is not None:
        unload_secondary_model()
        
    # If the model exists but is on CPU, unload it so we can reload on GPU
    if is_model_on_cpu():
        print(f"MixMod: Secondary model is on CPU, reloading it on GPU")
        unload_secondary_model()
    
    if not os.path.exists(model_path):
        print(f"MixMod: Model path {model_path} does not exist")
        return None
    
    try:
        print(f"MixMod: Loading secondary model from {model_path}")
        # Load state dict directly
        checkpoint_info = sd_models.CheckpointInfo(model_path)
        state_dict = load_torch_file(model_path)
        
        # Load model via forge_loader (same as in sd_models.py)
        secondary_model = forge_loader(state_dict)
        
        # Explicitly ensure the model is on GPU
        ensure_model_on_gpu()
        
        print(f"MixMod: Successfully loaded model from {model_path}")
        # Keep reference to prevent garbage collection
        return secondary_model
    except Exception as e:
        print(f"MixMod: Error loading model from {model_path}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return None


def unload_secondary_model():
    global secondary_model
    
    if secondary_model is not None:
        print(f"MixMod: Unloading secondary model to free VRAM")
        secondary_model = None
        # Force garbage collection to free up VRAM
        gc.collect()
        memory_management.soft_empty_cache()
        return True
    return False


class ModelAdapter:
    """Adapter to make the secondary model compatible with calc_cond_uncond_batch"""
    def __init__(self, model):
        self.model = model
        self.secondary_unet = model.forge_objects.unet.model
    
    def memory_required(self, shape):
        """Simple memory estimation"""
        return 1  # Return minimal memory to pass the check
    
    def apply_model(self, input_x, timestep, **kwargs):
        """Call the actual model's apply_model function"""
        # Ensure inputs and model are on the same device
        device = input_x.device
        
        # Double-check model is on the right device before running
        if next(self.secondary_unet.parameters()).device != device:
            print(f"MixMod: Device mismatch detected! Moving model to {device}")
            try:
                self.secondary_unet.to(device)
            except Exception as e:
                print(f"MixMod: Error moving model to match input device: {e}")
                traceback.print_exc()
                raise RuntimeError(f"MixMod: Failed to move model to {device}")
                
        # Ensure all inputs are on the same device
        timestep = timestep.to(device)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        v[k2] = v2.to(device)
        
        return self.secondary_unet.apply_model(input_x, timestep, **kwargs)


class MixModPatcher:
    def __init__(self):
        self.secondary_model = None
        self.secondary_model_path = None
        
    def patch(self, model, weight, cfg1, cfg2, model_path):
        # Check if the model path changed
        force_reload = self.secondary_model_path != model_path
        
        # Check if we need to load a new model
        if self.secondary_model is None or force_reload or is_model_on_cpu():
            # Load or reload the model, forcing a reload if it's on CPU
            self.secondary_model = load_secondary_model(model_path, force_reload=force_reload)
            self.secondary_model_path = model_path
            
        if self.secondary_model is None:
            print(f"MixMod: Failed to load model from {model_path}")
            return (model,)
        
        # Explicitly ensure model is on GPU
        ensure_model_on_gpu()
            
        # Clone the model to avoid affecting other extensions
        m = model.clone()

        
        
        # Store strong references to prevent garbage collection
        m.mixmod_secondary_model = self.secondary_model

        # Create adapter for the secondary model
        secondary_model_adapter = ModelAdapter(self.secondary_model)
        m.mixmod_secondary_model_adapter = secondary_model_adapter
        
        
        def mixmod_post_cfg_function(args):
            # Extract parameters
            denoised = args["denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_denoised = args["cond_denoised"]
            cond = args["cond"]
            uncond = args["uncond"] 
            sigma = args["sigma"]
            input_x = args["input"]
            model_options = args["model_options"]
            try:
                # Ensure model is on the right device before prediction
                device = input_x.device
                if next(m.mixmod_secondary_model.forge_objects.unet.model.parameters()).device != device:
                    print(f"MixMod: Device mismatch detected! Moving model to {device}")
                    m.mixmod_secondary_model.forge_objects.unet.model.to(device)


                
                # Get predictions from secondary model
                secondary_cond_pred, secondary_uncond_pred = calc_cond_uncond_batch(
                    m.mixmod_secondary_model_adapter,
                    cond, 
                    uncond, 
                    input_x, 
                    sigma, 
                    model_options
                )

               
                # Mix models according to formula
                # cfg_result = uncond*(1-weight) + uncond2*weight + (cond-uncond)*cfg_scale + (cond2-uncond2)*cfg2
                
                result = uncond_denoised * (1 - weight) + secondary_uncond_pred * weight + \
                        (cond_denoised - uncond_denoised) * cfg1 + \
                        (secondary_cond_pred - secondary_uncond_pred) * cfg2
                
                #print(f"MixMod: Mixed models with weight {weight} and cfg2 {cfg2}")
                return result
            except Exception as e:
                print(f"MixMod: Error during mixing: {str(e)}")
                traceback.print_exc()
                return denoised  # Return original result if there's an error
        
        # Register the post-cfg function on the cloned model
        m.set_model_sampler_post_cfg_function(mixmod_post_cfg_function, disable_cfg1_optimization=True)
        
        return (m,)


# Use a single instance to maintain state
opMixMod = MixModPatcher()


class MixMod(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "MixMod"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):

        # Get all checkpoint models from the model list
        model_files = []
        for key, checkpoint_info in sd_models.checkpoints_list.items():
            model_files.append(checkpoint_info.filename)
        
        # Sort the model list for better UI presentation
        model_files.sort(key=lambda x: os.path.basename(x).lower())
        
        
        with gr.Accordion(open=False, label=self.title()):
            
            mixmod_enabled = gr.Checkbox(label='Enabled', value=False)
            mixmod_model_path = gr.Dropdown(label='Model Path', choices=model_files)
            mixmod_strength = gr.Slider(label='Strength', value=0.5, minimum=0.0, maximum=1.0, step=0.01)
            mixmod_cfg_scale_2 = gr.Slider(label='CFG Scale 2', value=3, minimum=-10.0, maximum=20.0, step=0.5)
            
        return mixmod_enabled, mixmod_strength, mixmod_cfg_scale_2, mixmod_model_path

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        mixmod_enabled, mixmod_strength, mixmod_cfg_scale_2, mixmod_model_path = script_args
        if not mixmod_enabled or not mixmod_model_path:
            return
            
        if not os.path.exists(mixmod_model_path):
            print(f"MixMod: Model path {mixmod_model_path} does not exist")
            return
        
        
        p2 = p

        weight = float(mixmod_strength)
        cfg1 = p.cfg_scale
        cfg2 = float(mixmod_cfg_scale_2)

        p2.setup_conds()

        # Get current UNet and patch it
        unet = p.sd_model.forge_objects.unet
        patched_unet = opMixMod.patch(unet, weight,cfg1, cfg2, mixmod_model_path)[0]
        
        # Replace the original UNet with our patched version
        p.sd_model.forge_objects.unet = patched_unet
        
        # Store a reference to the patched UNet and opMixMod in p to prevent garbage collection
        p.mixmod_patched_unet = patched_unet
        p.mixmod_patcher = opMixMod
        
        # Add parameters to generation info
        p.extra_generation_params.update(dict(
            mixmod_enabled=mixmod_enabled,
            mixmod_strength=mixmod_strength,
            mixmod_cfg_scale_2=cfg2,
            mixmod_model=os.path.basename(mixmod_model_path)
        ))
        
        print(f"MixMod: Enabled with strength {weight} and CFG scale {cfg1} and CFG scale 2 {cfg2}")
        
        return
    
    def postprocess(self, p, processed, *args):
        """Called after the processing is done. Good place to unload the model."""
        global secondary_model
        
        if hasattr(p, 'mixmod_patcher') and p.mixmod_patcher is not None:
            print("MixMod: Cleaning up after sampling")
            # Clean up the references in p
            #p.mixmod_patched_unet = None
            #p.mixmod_patcher = None
            
            # Unload the secondary model
            #unload_secondary_model()
            
            # Also clear the reference in the patcher
            #opMixMod.secondary_model = None
            
            # Force garbage collection to free up VRAM
            #gc.collect()
            #memory_management.soft_empty_cache()
            
        return processed
    
    # Use process_before_every_sampling instead of before_every_sampling
    def before_every_sampling(self, p, *args, **kwargs):
        pass


