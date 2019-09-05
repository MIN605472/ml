function plotConfMat(varargin)
%PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
%   and precision normalized percentages
%
%   usage: 
%   PLOTCONFMAT(confmat) plots the confmat with integers 1 to n as class labels
%   PLOTCONFMAT(confmat, labels) plots the confmat with the specified labels
%
%   Vahe Tshitoyan
%   20/08/2017
%
%   Arguments
%   confmat:            a square confusion matrix
%   labels (optional):  vector of class labels

% number of arguments
switch (nargin)
    case 0
       confmat = 1;
       labels = {'1'};
    case 1
       confmat = varargin{1};
       labels = 1:size(confmat, 1);
    otherwise
       confmat = varargin{1};
       labels = varargin{2};
end

confmat(isnan(confmat))=0; % in case there are NaN elements
numlabels = size(confmat, 1); % number of labels

precision = diag(confmat)' ./ sum(confmat, 1) * 100;
recall = diag(confmat) ./ sum(confmat, 2) * 100;

% calculate the percentage accuracies
confpercent = 100*confmat./repmat(sum(confmat, 2),1,numlabels);
% plotting the colors
figure();
imagesc([confpercent (ones(numlabels, 1) * 255); ones(1, numlabels+1) * 255]);
title(sprintf('Accuracy: %.2f%%', 100*trace(confmat)/sum(confmat(:))));
ylabel('True Class'); xlabel('Predicted Class');

% set the colormap
colormap(flipud(gray));

% Create strings from the matrix values and remove spaces
textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d\n');
textStrings = strtrim(cellstr(textStrings));

precisionStrings = num2str(precision(:), '%.1f\n');
precisionStrings = strtrim(cellstr(precisionStrings));
recallStrings = num2str(recall(:), '%.1f\n');
recallStrings = strtrim(cellstr(recallStrings));

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:numlabels);
hStrings = text(x(:),y(:),textStrings(:), ...
    'HorizontalAlignment','center');
% Get the middle value of the color range
midValue = mean(get(gca,'CLim'));
for j = 1:numlabels
    t  = text(j, numlabels + 1, precisionStrings(j), 'HorizontalAlignment', 'center');
    set(t,{'Color'}, {[1 1 1]});
    t = text(numlabels + 1, j, recallStrings(j), 'HorizontalAlignment', 'center');
    set(t,{'Color'}, {[1 1 1]});
end


% Choose white or black for the text color of the strings so
% they can be easily seen over the background color
textColors = repmat(confpercent(:) > midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));



% Setting the axis labels
yTickLabel = mat2cell(labels, 1, ones(1, length(labels)));
yTickLabel{length(yTickLabel) + 1} = 'precision';

xTickLabel = mat2cell(labels, 1, ones(1, length(labels)));
xTickLabel{length(xTickLabel) + 1} = 'recall';

set(gca,'XTick',1:numlabels+1,...
    'XTickLabel',xTickLabel,...
    'YTick',1:numlabels+1,...
    'YTickLabel', yTickLabel,...
    'TickLength',[0 0]);
end