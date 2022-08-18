new_array = readmatrix('results/20220818_1252/accuracy.csv');

iterations = new_array(:,1)';
train_loss = new_array(:,2)';
train_acc = new_array(:,3)';
valid_acc = new_array(:,4)';

figure(1);clf;
subplot(2,1,1);
semilogy(iterations,train_loss,'LineWidth',1.7);
ylabel('Loss');
ax = gca; 
ax.FontSize = 13; 

subplot(2,1,2);
plot(iterations,train_acc,'LineWidth',1.7);
hold on
plot(iterations,valid_acc,'LineWidth',1.7);
legend({'Train','Valid'},'Location','northwest');
xlabel('Epoch');
ylabel('Accuracy');
ax = gca; 
ax.FontSize = 13; 