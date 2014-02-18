img = imread("1.png");
block0 = img(1:32, 1:32);
block30 = img(481:512, 481:512);

block0_flat = single(reshape(block0, 1, 32*32));
block30_flat = single(reshape(block30, 1, 32*32));

block0_flat_sorted = sort(block0_flat);
block30_flat_sorted = sort(block30_flat);

%mean(mean(block0))
var(block0_flat) % should return 2033.33
