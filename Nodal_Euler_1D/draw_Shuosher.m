A=textread('data.txt');
xx=-5+10/2000/2:10/2000:5-10/2000/2;

figure(1); hold on;
plot(xx,A,'k-','LineWidth',1.2);
plot(Xc,Q1,'b--','LineWidth',1);
legend('WENO-Z 5000','DG 4096')
