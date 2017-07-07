function draw_confusion_table

% load classes
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
fclose(fid);
classes = cell(22, 1);
classes{1} = 'background';
classes(2:end) = C{1};

for i = 2:numel(classes)
    classes{i} = classes{i}(5:end);
    classes{i} = strrep(classes{i}, '_', ' ');
end 

% load confusion matrix
cmatrix = load('confusion_matrix.txt');

CTableProb = draw_ctable(cmatrix, 1, classes);
set(gca,'XTickLabelRotation',-45)