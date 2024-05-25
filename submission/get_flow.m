function [F, M] = get_flow(netw)
% Returns the matrix of fluxes M and flow potentials F
% An alternative of the following code is to compute the Laplacian matrix
% using the built-in L = laplacian(G) Matlab function. However, for small networks
% (i.e. N0 < 1600), this code seems faster.
% (need further investigations for assessing runtime differences).

%Preallocs:
sized = size(netw.graph.Nodes,1);

%for Matlab 'graph' adaptation, just re-create the adjacency matrix:
netw.adjm = full(adjacency(netw.graph));

%use MATLAB version:
[P,S] = prepare_linsys(netw, sized);

%Remove both inflow and outflow entries from P:
P([netw.inflow, netw.outflow],:) = [];
P(:,[netw.inflow, netw.outflow]) = [];
S([netw.inflow, netw.outflow]) = [];

%Solve the linear system:
FF = P\S;

%Re-insert inflow and outflow nodes:
F(1:min([netw.inflow, netw.outflow])-1) = FF(1:min([netw.inflow, netw.outflow])-1);
F(min([netw.inflow, netw.outflow])+1:max([netw.inflow, netw.outflow])-1) = FF(min([netw.inflow, netw.outflow]):max([netw.inflow, netw.outflow])-2);
F(max([netw.inflow, netw.outflow])+1:size(netw.adjm,1)) = FF(max([netw.inflow, netw.outflow])-1:size(netw.adjm,1)-2);
%Insert:
if(min([netw.inflow, netw.outflow]) == netw.inflow)
    F(min([netw.inflow, netw.outflow]))=1;
    F(max([netw.inflow, netw.outflow]))=0;
else
    F(min([netw.inflow, netw.outflow]))=0;
    F(max([netw.inflow, netw.outflow]))=1;
end
F = F'; %this can take a long time, depending on the network's size

%Then compute the fluxes matrix M, according to F (flow potentials):
M = compute_q(netw, sized, F);


function M = compute_q(netw, sized, F)
% Prealloc (warning: because 0 can possibly be a minimum value, but in
% remove_weakest_link() we actually do: min(netw.fluxes(netw.fluxes > 0)))
M = zeros(sized, sized); 

for i=1:size(netw.adjm,1)
    idx = netw.adjm(i,:) == 1;          %for each connected nodes
    D = netw.dis_eucl(i, idx);          %gather the distances
    M(i, idx) = -(F(i) - F(idx)) ./ D'; %then compute the associated flow
end

function [P, S] = prepare_linsys(netw, sized)
%prepare Linear system for solving:

%Prealloc:
P = zeros(sized, sized); %Matrix of Pi's (without inflow and outflow entries)
S = zeros(sized, 1);  %Conservation of mass, thus outputs are 0's

for i=1:size(netw.adjm,1)
    idx = netw.adjm(i,:) == 1;   %gather indexes of connected nodes.
    D = netw.dis_eucl(i, idx);   %gather corresponding euclidean distances

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Standard Node (non inflow, non outflow)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Check if there is link to the inflow node:
    if(any(find(idx) == netw.inflow))
        D2 = D; %save D.
        S(i) = -1/D(find(idx) == netw.inflow); %alter output accordingly (we know inflow value = 1)
        D(find(idx) == netw.inflow) = []; %remove inflow from distances
        idx(netw.inflow) = 0; %then erase coefficient from idx
        %Finally, perform standard computation
        P(i,i) = -sum(1./D2);
    else
        %Standard computation (ie. current node 'i' is not linked to inflow):
        P(i,i) = -sum(1./D);
    end
    
    %Assign coefficients values to connected nodes:
    P(i,idx) = 1./D;
end

