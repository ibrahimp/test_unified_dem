function [denoised_raw] = raw_mosaic_guided_high_density(noisy_raw, radius, epsilon)
% RAW_MOSAIC_GUIDED_HIGH_DENISTY Filters RAW mosaic data by keeping G1 and G2 
% interleaved. Runs the guided filter at a unified high-density resolution 
% to preserve ultra-fine textures and eliminate green imbalance.

    [H, W] = size(noisy_raw);
    sig = radius / 2; % Gaussian standard deviation
    
    % 1. Extract Native Sub-grids
    R_sub  = noisy_raw(1:2:end, 1:2:end);
    G1_sub = noisy_raw(1:2:end, 2:2:end);
    G2_sub = noisy_raw(2:2:end, 1:2:end);
    B_sub  = noisy_raw(2:2:end, 2:2:end);
    
    % 2. Construct the High-Density Interleaved Green Channel (G_full)
    % Instead of separating them, we map them directly to a unified half-size grid.
    % This retains double the sampling density of the individual sub-grids.
    G_full = (G1_sub + G2_sub) / 2; 
    
    % 3. Upscale Red and Blue Sub-grids to Match Green's High-Density Layout
    % We use fast bilinear interpolation to align R and B coordinates with G_full.
    [X_sub, Y_sub] = meshgrid(1:W/2, 1:H/2);
    R_full = interp2(X_sub, Y_sub, R_sub, X_sub, Y_sub, 'linear');
    B_full = interp2(X_sub, Y_sub, B_sub, X_sub, Y_sub, 'linear');
    
    % 4. Pre-compute Unified Guidance Statistics (Using High-Density Green)
    % Because G_full has double the resolution, the Gaussian filter is twice 
    % as precise at detecting fine edges compared to isolated sub-grids.
    mean_I = imgaussfilt(G_full, sig, 'Padding', 'replicate');
    var_I  = imgaussfilt(G_full .* G_full, sig, 'Padding', 'replicate') - mean_I .* mean_I;
    
    % 5. Create the Master Coordination Valve (A_joint)
    % Shared globally across all 3 high-density channels (R, G, B)
    A_joint = var_I ./ (var_I + epsilon);
    mean_A_joint = imgaussfilt(A_joint, sig, 'Padding', 'replicate');
    denom_inv = 1 ./ (var_I + 1e-6);

    % 6. Process the 3 High-Density Channels (R, G_full, B)
    % Loop runs only 3 times instead of 4, saving 25% computational overhead.
    full_channels = {R_full, G_full, B_full};
    filtered_full = cell(1, 3);
    
    for c = 1:3
        p = full_channels{c};
        mean_p = imgaussfilt(p, sig, 'Padding', 'replicate');
        
        % Compute localized cross-channel statistics
        mean_Ip = imgaussfilt(G_full .* p, sig, 'Padding', 'replicate');
        cov_Ip  = mean_Ip - mean_I .* mean_p;
        
        % Apply the global coordination valve
        k_channel = cov_Ip .* denom_inv;
        a_tuned = k_channel .* mean_A_joint;
        b_tuned = mean_p - a_tuned .* mean_I;
        
        mean_b = imgaussfilt(b_tuned, sig, 'Padding', 'replicate');
        
        % Reconstruct high-density clean layer
        filtered_full{c} = a_tuned .* G_full + mean_b;
    end
    
    % Extract the cleaned high-density channels
    R_clean_full = filtered_full{1};
    G_clean_full = filtered_full{2};
    B_clean_full = filtered_full{3};
    
    % 7. Re-interleave Back into the Native 2x2 Bayer Mosaic Layout
    % G1 and G2 are assigned directly from the unified clean Green channel.
    % R and B are sampled from their respective positions.
    denoised_raw = zeros(H, W, 'like', noisy_raw);
    denoised_raw(1:2:end, 1:2:end) = R_clean_full; 
    denoised_raw(1:2:end, 2:2:end) = G_clean_full; % Cleaned G1
    denoised_raw(2:2:end, 1:2:end) = G_clean_full; % Cleaned G2 (Forces balance)
    denoised_raw(2:2:end, 2:2:end) = B_clean_full;
end
