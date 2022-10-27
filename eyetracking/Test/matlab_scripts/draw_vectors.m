function draw_vectors(vectors)
    % print all ten predicted vectors
    % and their mean vector
    v = mean(vectors);
%     q = quiver3(0,0,0,v(1),v(3),-v(2),'r');
    
    for i = 1:size(vectors,1)
        q = quiver3(0,0,0,vectors(i,1),vectors(i,3),-vectors(i,2),'c');
        q.LineWidth = 1;
        hold on
    end
end
