args = argv();
if numel(args) ~= 10
    error(['usage: run_reference.m REFERENCE_ROOT INPUT OUTPUT WIDTH HEIGHT ' ...
           'PATTERN MODE EXPECTED_MAIN_SHA256 EXPECTED_README_SHA256 EXPECTED_ARCHIVE_SHA256']);
end

pkg load image;
reference_root = args{1};
input_file = args{2};
output_file = args{3};
width = str2double(args{4});
height = str2double(args{5});
pattern = args{6};
mode = args{7};
expected_main_sha256 = args{8};
expected_readme_sha256 = args{9};
expected_archive_sha256 = args{10};

if ~any(strcmp(pattern, {'grbg', 'rggb', 'gbrg', 'bggr'}))
    error('unsupported Bayer pattern');
end
if ~any(strcmp(mode, {'green', 'full'}))
    error('unsupported ARI reference mode');
end

function authenticate(path, expected, label)
    quoted = strrep(path, '''', '''\''''');
    [status, output] = system(sprintf('sha256sum ''%s''', quoted));
    if status ~= 0
        error(['cannot hash ' label]);
    end
    actual = strtok(output);
    if ~strcmp(actual, expected)
        error([label ' SHA-256 mismatch']);
    end
end

authenticate(fullfile(reference_root, 'demosaic_ARI.m'), expected_main_sha256, ...
             'ARI main reference');
authenticate(fullfile(reference_root, 'readme.txt'), expected_readme_sha256, ...
             'ARI readme');
archive_path = getenv('ARI_REFERENCE_ARCHIVE');
if isempty(archive_path)
    error('ARI_REFERENCE_ARCHIVE is required');
end
authenticate(archive_path, expected_archive_sha256, 'ARI archive');

addpath(reference_root);
input = fopen(input_file, 'rb', 'ieee-le');
if input < 0
    error('cannot open ARI reference input');
end
rgb = zeros(height, width, 3);
for channel = 1:3
    values = fread(input, width * height, 'single=>double');
    if numel(values) ~= width * height
        fclose(input);
        error('incorrect ARI reference input size');
    end
    rgb(:,:,channel) = reshape(values, [width, height])';
end
if fread(input, 1, 'uint8')
    fclose(input);
    error('extended ARI reference input');
end
fclose(input);

% The reviewed demonstration converts source images to double in the native
% 0..255 domain before calling demosaic_ARI().  Green-only mode calls the
% exact first stage so degenerate constant scenes do not enter Octave's
% complex-number-incompatible diagonal filtering path.
if strcmp(mode, 'green')
    [mosaic, mask] = mosaic_bayer(rgb .* 255.0, pattern);
    result = green_interpolation(mosaic, mask, pattern, 1e-10) ./ 255.0;
else
    result = demosaic_ARI(rgb .* 255.0, pattern) ./ 255.0;
end
if any(~isfinite(result(:)))
    error('ARI reference produced a non-finite value');
end

output = fopen(output_file, 'wb', 'ieee-le');
if output < 0
    error('cannot create ARI reference output');
end
if strcmp(mode, 'green')
    fwrite(output, single(result)', 'single');
else
    for channel = 1:3
        fwrite(output, single(result(:,:,channel))', 'single');
    end
end
fclose(output);
