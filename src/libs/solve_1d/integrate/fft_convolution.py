import torch 
import torch.fft as fft

def conv1d(f, 
           kernel_list, 
           dx, 
           Nx, 
           kernel_preprocessed=True, 
           verbose=True,
           ):
    start = Nx//2 + 1

    if kernel_preprocessed:
        kernel_list_fft = kernel_list

    else:
        if verbose: print("WARNING: Computing kernel fft during runtime")
        kernel_list_fft = []
        for k_i in kernel_list:
            kernel_list_fft.append(
                torch.fft.rfft(k_i)
            )
    
    f_fft = torch.fft.rfft(f)

    conv = []
    for k_i_fft in kernel_list_fft:
        fft_conv_i = f_fft * k_i_fft  # (BS, Nx//2), (1, Nx//2)
        conv_i = torch.roll(torch.fft.irfft(fft_conv_i) * dx, start, dims = 1)
        conv.append(conv_i)

    return conv  # len(k) x (BS, Nx//2)