function rate=knnClassification(k,trainData,trainLabel,testData,testLabel)
%trainData:  每一行表示一个数据点
%testData:   每一行表示一个数据点
%trainLabel: 行向量
%testLabel:  行向量
%k:           k近邻个数
%rate:        识别准确率

trainData=double(trainData); 
testData=double(testData);

[m1,n1]=size(trainData);
[m2,n2]=size(testData);

distance=L2_distance2(trainData',testData',0);
%distance=zeros(n1,n2);
%for i=1:n2
%    for j=1:n1
%        distance(j,i)=sqrt(sum((trainData(:,j)-testData(:,i)).^2));
%    end
%end

%distance=zeros(n1,n2);
%for i=1:n2
%    for j=1:n1
%        distance(j,i)=dot(trainData(:,j),testData(:,i))/(norm(trainData(:,j))*norm(testData(:,i)));
%    end
%end

[~,order]=sort(distance,1);

label1=zeros(k,m2);
for i=1:m2
    label1(:,i)=trainLabel(order(1:k,i));
end
resultLabel=mode(label1);
resultLabel = resultLabel';
len_predict = length(find(resultLabel==testLabel));

rate=len_predict/length(testLabel);





