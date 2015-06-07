x = [0:0,05:20];


for i = 1:2:10
    y = i * round(x/i);
    hold on
    plot(x, y)
end
