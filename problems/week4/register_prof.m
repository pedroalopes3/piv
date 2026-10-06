im1=imread('parede1.jpg');
im2=imread('parede2.jpg');

%%
close all;
figure(1);
imshow(im1)
fprintf('Click on the poster corners in a fixed order \n')
[u1 v1]=ginput(4)
hold
plot(u1,v1,'*')
hold off;
fprintf('Click on the poster corners in THE SAME order \n')
figure(2);
imshow(im2)
[u2 v2]=ginput(4)
hold on
plot(u2,v2,'*')
hold off

%%
t=cp2tform([u2 v2],[u1 v1],'projective');
im3=imtransform(im2,t);
figure(4);
imagesc(im3)
imt1=imtransform(im1,t,'Xdata',[-300 800],'YData',[-250 600]);

%%
t2=cp2tform([u1 v1],[u1 v1],'projective');
t2
t2.tdata
t2.tdata.T
%pts=[0 0; 200 0;200 300;0 300]
pts=[0 0; 200 0;200 sqrt(2)*200;0 sqrt(2)*200]
[u1 v1]
t1=cp2tform([u1 v1],pts,'projective');
imt1=imtransform(im1,t1);
figure;
imagesc(imt1)
t2=cp2tform([u2 v2],pts,'projective');
imt2=imtransform(im2,t2);
figure
imagesc(imt2)

%imt2=imtransform(im2,t2,'Xdata',[-200 800],'YData',[-50 800]);
%imagesc(imt2)
%imt2=imtransform(im2,t2,'Xdata',[-200 800],'YData',[-250 600]);
%imagesc(imt2)
%imt2=imtransform(im2,t2,'Xdata',[-300 800],'YData',[-250 600]);
%imagesc(imt2)

imt2=imtransform(im2,t2,'Xdata',[-400 800],'YData',[-250 600]);
imagesc(imt2)
imt1=imtransform(im1,t1,'Xdata',[-400 800],'YData',[-250 600]);
imagesc(imt1)
figure

%imagesc(double(imt1)*.5+double(imt2)*.5)
imagesc(uint8(double(imt1)*.5+double(imt2)*.5))