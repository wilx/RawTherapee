function o = imgconv2(img, a)
	% The double rotation (here and inside conv2) makes this a same-sized
	% zero-padded correlation with a.  The C++ reference helper deliberately
	% preserves that convention, including asymmetric kernels.
	o = conv2(img, rot90(a,2), 'same');
end
