opts = detectImportOptions('results\jungwoo\accuracy.csv');
raw_matrix = readtable('results\jungwoo\accuracy.csv',opts);
row_count = size(raw_matrix,1);

new_matrix = raw_matrix;
for it = 1:row_count
    text3 = raw_matrix{it,3}{1};
    new_matrix{it,3}{1} = str2double(text3(8:13));
    %text5 = raw_matrix{it,5}{1};
    %new_matrix{it,5}{1} = str2double(text5(8:13));
end

writetable(new_matrix,'results\jungwoo\fixed_csv_new.csv');