args = argv();
if (numel(args) ~= 7) && (numel(args) ~= 8)
    error(['usage: run_reference.m MAIN INPUT OUTPUT_PREFIX WIDTH HEIGHT ' ...
           'EXPECTED_SHA256 MAX_SLOW [MIN_SLOW]']);
end

pkg load image;
main_file = args{1};
input_file = args{2};
output_prefix = args{3};
width = str2double(args{4});
height = str2double(args{5});
expected_sha256 = args{6};
max_slow = str2double(args{7});
min_slow = 0;
if numel(args) == 8
    min_slow = str2double(args{8});
end
if ~isfinite(min_slow) || ~isfinite(max_slow) || min_slow < 0 || ...
        max_slow < min_slow || min_slow ~= fix(min_slow) || max_slow ~= fix(max_slow)
    error('MIN_SLOW and MAX_SLOW must define a nonnegative integer range');
end

[status, actual_sha256] = system(sprintf('sha256sum %s', ...
    strrep(main_file, '''', '''\''''')));
if status ~= 0
    error('cannot hash the ULRI X-Trans reference');
end
actual_sha256 = strtok(actual_sha256);
if ~strcmp(actual_sha256, expected_sha256)
    error('ULRI X-Trans reference SHA-256 mismatch');
end

addpath(fileparts(main_file));
input = fopen(input_file, 'rb', 'ieee-le');
if input < 0
    error('cannot open ULRI reference input');
end
mosaic = reshape(fread(input, width * height, 'single=>single'), [width, height])';
fclose(input);
mosaic = uint16(mosaic .* single(65535));

for slow = min_slow:max_slow
    % Use the authenticated uint16 branch.  The published floating-input type
    % probe uses max(...,'all'), which GNU Octave 8 does not implement.  The
    % corpus mosaic is explicitly quantized before this script, so this does
    % not introduce a hidden input conversion.
    rgb = single(function_demosaic_x_trans(mosaic, slow, 2, 1E-2));
    rgb = rgb ./ single(65535);
    output_file = sprintf('%s-slow%d.f32le', output_prefix, slow);
    output = fopen(output_file, 'wb', 'ieee-le');
    if output < 0
        error('cannot create ULRI reference output');
    end
    for channel = 1:3
        fwrite(output, rgb(:,:,channel)', 'single');
    end
    fclose(output);
end
