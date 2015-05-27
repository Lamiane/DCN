x = [-100:3:100];
y = [-100:3:100];
[xx, yy] = meshgrid(x, y);
zz = max(xx, yy);
surf(xx, yy, zz)
