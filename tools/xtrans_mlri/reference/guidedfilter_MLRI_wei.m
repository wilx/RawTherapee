function q = guidedfilter_MLRI_wei(G, R, mask, I, p, M, h, v, eps)
	% The number of the sampled pixels in each local patch
	filterSize = single(ones([2*v+1, 2*h+1]));
	N = imgconv2(M, filterSize);
	N(N == 0) = 1;

	% The size of each local patch; N=(2r+1)^2 except for boundary pixels.
	mean_Ip = imgconv2(I.*p.*M, filterSize) ./ N;
	mean_II = imgconv2(I.*I.*M, filterSize) ./ N;

	% Eqn. (5) in the SPIE paper.
	a = mean_Ip ./ (mean_II + eps);
	clear mean_Ip mean_II N;

	% Eqn. (6) in the SPIE paper.
	N3 = imgconv2(mask, filterSize);
	N3(N3 == 0) = 1;

	mG = imgconv2(G.*mask, filterSize);
	mR = imgconv2(R.*mask, filterSize);

	mean_G = mG ./ N3;
	mean_R = mR ./ N3;
	b = mean_R - a .* mean_G;

	% Weighted average added in the expanded TIP paper.
	dif = (b.*b).*N3 + (a.*b*2).*mG - (2*b).*mR - ...
		  imgconv2(R.*G.*mask,filterSize).*(2*a) + ...
		  imgconv2(G.*G.*mask,filterSize).*(a.*a) + ...
		  imgconv2(R.*R.*mask,filterSize);

	dif = dif ./ N3;
	clear mG mR mean_R mean_G N3;

	dif( dif < 0.01 ) = 0.01;
	dif = 1./dif;
	wdif = imgconv2(dif,filterSize);
	wdif( wdif < 0.01 ) = 0.01;
	mean_a = imgconv2(a.*dif,filterSize) ./ wdif;
	mean_b = imgconv2(b.*dif,filterSize) ./ wdif;
	clear dif wdif a b;

	% Eqn. (8) in the expanded TIP paper.
	q = mean_a .* G + mean_b;
end
