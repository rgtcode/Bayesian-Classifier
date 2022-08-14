function [m,C] = GetMeanCov(data, labels, classID)

    [~,ddim] = size(data);

    % Get data for particulat classID
    classSamplesInd = find(labels==classID);
    classData   = data(classSamplesInd, :);
    classLabels = labels(classSamplesInd);
    
    % Compute mean vector
    for i = 1:ddim
       m(i) = GetMean(classData(:,i)); 
    end
    
    % Compute Covariance matrix - need to compute only unique elements of
    % this symmetric matric
    for i = 1:ddim
        for j = 1:ddim
            v1 = classData(:,i);
            v2 = classData(:,j);
            v1v2 = v1.*v2;
            m_v1 = GetMean(v1);
            m_v2 = GetMean(v2);
            m_v1v2 = GetMean(v1v2);
            C(i,j) = m_v1v2 - m_v1*m_v2;
        end
    end
end