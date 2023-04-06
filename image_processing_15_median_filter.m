clc;
clear all;
clear;
close all;

I=imread("cameraman.tif");
figure, imshow(uint8(I))

%Inoisy=imnoise(I,"gaussian",0.1);
Inoisy=imnoise(I,"salt & pepper",0.1);
figure,imshow(uint8(Inoisy)),title("Noisy")

[h,w]=size(I);
I2=zeros(h,w);

k=5;
fkh=floor(k/2);
fkw=floor(k/2);

for i=fkh+1:h-fkh
    for j=fkw+1:w-fkw
        block=Inoisy(i-fkh:i+fkh,j-fkw:j+fkw);
        block_1d=reshape(block,1,k*k);
        block_1d_sort=sort(block_1d);
        
        if (Inoisy(i,j)==0||Inoisy(i,j)==255)
            I2(i,j)=block_1d_sort(ceil((k*k)/2));
        else
            I2(i,j)=Inoisy(i,j);
        end
    end
end

figure,imshow(uint8(I2)),title("Noisy")

