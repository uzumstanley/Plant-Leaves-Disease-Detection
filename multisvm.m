function result = multisvm(TrainingData,GroupTrain,TestData)
u=unique(GroupTrain);
numClasses=length(u);
result = zeros(size(TestData,1),1);
for k=1:numClasses
    GroupTrainNew=(GroupTrain==u(k));
    svmModel = fitcsvm(TrainingData,GroupTrainNew,'KernelFunction','linear');
    result(:,k) = predict(svmModel,TestData);
end
[~,result] = max(result,[],2);
result = u(result);
