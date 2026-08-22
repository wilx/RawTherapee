args = argv();
if numel(args) ~= 6
    error('usage: run_reference.m MAIN INPUT OUTPUT WIDTH HEIGHT EXPECTED_SHA256');
end

pkg load image;
addpath(fileparts(mfilename('fullpath')));
main_file = args{1};
input_file = args{2};
output_file = args{3};
width = str2double(args{4});
height = str2double(args{5});
expected_sha256 = args{6};

[status, actual_sha256] = system(sprintf('sha256sum %s', ...
    strrep(main_file, '''', '''\''''')));
if status ~= 0
    error('cannot hash the X-Trans MATLAB reference');
end
actual_sha256 = strtok(actual_sha256);
if ~strcmp(actual_sha256, expected_sha256)
    error('X-Trans MATLAB reference SHA-256 mismatch');
end

addpath(fileparts(main_file));
input = fopen(input_file, 'rb', 'ieee-le');
if input < 0
    error('cannot open golden input');
end
mosaic = reshape(fread(input, width * height, 'single=>single'), [width, height])';
fclose(input);

% Exercise the authenticated implementation's uint16 branch.  Convert the
% returned integer samples back to single only for a language-neutral corpus.
rgb = single(function_demosaic_x_trans(uint16(mosaic), 1, 2, 1E-2));

output = fopen(output_file, 'wb', 'ieee-le');
if output < 0
    error('cannot create golden output');
end
for channel = 1:3
    fwrite(output, rgb(:,:,channel)', 'single');
end
fclose(output);
