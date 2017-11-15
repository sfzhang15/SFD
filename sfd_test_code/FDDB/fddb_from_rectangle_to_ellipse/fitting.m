close all; clear; clc;
load Fit.mat

%% from rectangle to ellipse
fout=fopen('../sfd_fddb_dets_fit.txt','w');
fid=fopen('../sfd_fddb_dets.txt','r');
while ~feof(fid)
    imName=fgetl(fid);
    fprintf(fout,imName);
    fprintf(fout,'\n');
    numFaces=str2num(fgetl(fid));
    fprintf(fout,'%d\n',numFaces);
    for j=1:numFaces
        rect=fgetl(fid);
        rect=str2num(rect);
        score=rect(5);
        temp=rect(1:4)-meanX;
        temp=temp./stdX;
        ellipse=temp*w;
        ellipse=ellipse.*stdY;
        ellipse=ellipse+meanY;
        if ellipse(3)>=0
            ellipse(3)=-pi/2+ellipse(3);
        else
            ellipse(3)=ellipse(3)+pi/2;
        end
        fprintf(fout,'%f %f %f %f %f %f\n',ellipse(1),ellipse(2),ellipse(3),ellipse(4),ellipse(5),score);
    end
end
fclose(fid);
fclose(fout);
