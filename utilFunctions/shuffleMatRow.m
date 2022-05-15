function ret = shuffleMatRow(mat)

[r c] = size(mat);
shuffledRow = randperm(r);
ret = mat(shuffledRow, :);
end
