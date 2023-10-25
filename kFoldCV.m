function indices = kFoldCV(yapp,cross_num, nClasses)
flag = 1;
while flag
%	cross_num = 10;
	indices = crossvalind('Kfold',length(yapp), cross_num);
	for cros = 1:cross_num
		test_indx = (indices == cros);
		test_label = yapp(test_indx);
		if length(unique(test_label)) ~= nClasses
			flag = 1;
			break;
		else
			flag = 0;
		end
	end
end