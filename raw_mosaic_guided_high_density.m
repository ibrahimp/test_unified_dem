function [denoised_raw] = raw_mosaic_guided_high_density(noisy_raw, radius, sigma, epsilon)
% RAW_MOSAIC_GUIDED_UNPAIRED Handles GRBG Bayer layout with fully UNPAIRED
% evaluation radius and Gaussian smoothing sigma parameters.
%
% Inputs:
%   noisy_raw : 2D Matrix (Double) - Normalized linear RAW data (0.0 to 1.0)
%   radius    : Integer            - Statistical window radius (e.g., 4 for a 9x9 box context)
%   sigma     : Double             - Gaussian filter standard deviation for edge smoothness
%   epsilon   : Double             - Regularization noise gate threshold

    [H, W] = size(noisy_raw);
    window_size = 2 * radius + 1;
    
    % 1. Separate native channels based on GRBG layout
    G1_sub = noisy_raw(1:2:end, 1:2:end);
    R_sub  = noisy_raw(1:2:end, 2:2:end);
    B_sub  = noisy_raw(2:2:end, 1:2:end);
    G2_sub = noisy_raw(2:2:end, 2:2:end);
    
    % 2. Construct Full-Resolution Green Guidance Map (H x W)
    I_G = zeros(H, W, 'like', noisy_raw);
    I_G(1:2:end, 1:2:end) = G1_sub;
    I_G(2:2:end, 2:2:end) = G2_sub;
    
    diamond_kernel = [0 0.25 0; 0.25 0 0.25; 0 0.25 0];
    G_interpolated = imfilter(I_G, diamond_kernel, 'replicate');
    
    mask_G = zeros(H, W);
    mask_G(1:2:end, 1:2:end) = 1;
    mask_G(2:2:end, 2:2:end) = 1;
    I_G = I_G .* mask_G + G_interpolated .* (1 - mask_G);
    
    % 3. Upscale Red and Blue to Full H x W Resolution
    [X_sub_R, Y_sub_R] = meshgrid(2:2:W, 1:2:H);
    [X_full, Y_full] = meshgrid(1:W, 1:H);
    R_full = interp2(X_sub_R, Y_sub_R, R_sub, X_full, Y_full, 'linear');
    
    [X_sub_B, Y_sub_B] = meshgrid(1:2:W, 2:2:H);
    B_full = interp2(X_sub_B, Y_sub_B, B_sub, X_full, Y_full, 'linear');
    
    % 4. Pre-compute Guidance Statistics using UNPAIRED parameters
    % Use imboxfilt for the neighborhood window bounds, imgaussfilt for spatial smoothing
    mean_I = imboxfilt(I_G, window_size, 'Padding', 'replicate');
    var_I  = imboxfilt(I_G .* I_G, window_size, 'Padding', 'replicate') - mean_I .* mean_I;
    
    % 5. Create and Smooth the Master Coordination Valve (A_joint)
    A_joint = var_I ./ (var_I + epsilon);
    
    % The smoothing of the edge valve is where SIGMA dictates the edge falloff transition
    mean_A_joint = imgaussfilt(A_joint, sigma, 'Padding', 'replicate');
    denom_inv = 1 ./ (var_I + 1e-6);

    % 6. Process the 3 Full-Resolution Channels
    full_channels = {R_full, I_G, B_full};
    filtered_full = cell(1, 3);
    
    for c = 1:3
        p = full_channels{c};
        mean_p = imboxfilt(p, window_size, 'Padding', 'replicate');
        
        mean_Ip = imboxfilt(I_G .* p, window_size, 'Padding', 'replicate');
        cov_Ip  = mean_Ip - mean_I .* mean_p;
        
        k_channel = cov_Ip .* denom_inv;
        a_tuned = k_channel .* mean_A_joint;
        b_tuned = mean_p - a_tuned .* mean_I;
        
        % Smooth the offset factor with the independent Gaussian sigma
        mean_b = imgaussfilt(b_tuned, sigma, 'Padding', 'replicate');
        filtered_full{c} = a_tuned .* I_G + mean_b;
    end
    
    % Extract the cleaned layers
    R_clean = filtered_full{1};
    G_clean = filtered_full{2};
    B_clean = filtered_full{3};
    
    % 7. Re-sample Back into native GRBG Mosaic Layout
    denoised_raw = zeros(H, W, 'like', noisy_raw);
    denoised_raw(1:2:end, 1:2:end) = G_clean(1:2:end, 1:2:end); 
    denoised_raw(1:2:end, 2:2:end) = R_clean(1:2:end, 2:2:end); 
    denoised_raw(2:2:end, 1:2:end) = B_clean(2:2:end, 1:2:end);
    denoised_raw(2:2:end, 2:2:end) = G_clean(2:2:end, 2:2:end); 
end
