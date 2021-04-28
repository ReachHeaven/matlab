inData = inData(1:4000);
outData = outData(1:4000);
TrainDatainput = inData(1:2500);
TrainTestinput = inData(2501:4000);
TrainDataoutput = outData(1:2500);
TrainTestoutput = outData(2501:4000);
time = 1:4000;
plot(time(1:2500), TrainDatainput, time(2501:end), TrainTestinput)
plot(time(1:2500), TrainDataoutput, time(2501:end), TrainTestoutput)

TDI = con2seq(TrainDatainput');
TDO = con2seq(TrainDataoutput');
TTI = con2seq(TrainTestinput');
TTO = con2seq(TrainTestoutput');

rng(5)

net = timedelaynet(1:4,7);%ïðè äàííûõ  ïàðàìåòðàõ, ïîëó÷åíà ñàìàÿ íèçêàÿ ïîãðåøíîñòü
net.divideFcn="divideblock";

[Xs,Xi,Ai,Ts] = preparets(net,TDI,TDO);
net = train(net,Xs,Ts,Xi,Ai);
view(net)

R = net(TDI);

plot(cell2mat(TDO))
hold on
plot(cell2mat(R))


Q = net(TTI);
plot(cell2mat(TTO))%Äëÿ ñðàâíåíèÿ ñ òåñòîâûìè äàííûìè
hold on
plot(cell2mat(Q))



[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y);
[netc,Xic,Aic] = closeloop(net,Xf,Af);
view(netc)
y2 = netc(in,Xic,Aic);
view(net)
net = train(net,Xs,Ts,Xi,Ai);
