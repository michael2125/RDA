%% d = L2_distance(a,b,df)
% compute the L2 norm of a and b.
% eg.
% a=[3,4,5,6,7,8;3,4,5,2,1,8]
% a =
%      3     4     5     6     7     8
%      3     4     5     2     1     8
% d=L2_distance(a,a,1)
% d =
%          0    1.4142    2.8284    3.1623    4.4721    7.0711
%     1.4142         0    1.4142    2.8284    4.2426    5.6569
%     2.8284    1.4142         0    3.1623    4.4721    4.2426
%     3.1623    2.8284    3.1623         0    1.4142    6.3246
%     4.4721    4.2426    4.4721    1.4142         0    7.0711
%     7.0711    5.6569    4.2426    6.3246    7.0711         0
function d = L2_distance(a,b,df)
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end
aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);
d = real(d); 
if (df==1)
  d = d.*(1-eye(size(d)));
end