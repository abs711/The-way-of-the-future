function [p1, p2, CI] = TOST(sample1, sample2, d1, d2, alpha)
%Two One-Sided Test for Equivalence (as per Roger et al, 1993)

%This function tests if two samples come from distributions with different 
%means, against the alternative hypothesis that the means are the same.
%I.e.,
%H0:  the two samples have different means (the difference between the
%means falls outside of the equivalence interval [d1, d2])
%HA:  the two samples have equivalent means (the difference between the 
%means falls within the equivalence interval [d1, d2])
%The null hypothesis is rejected if max([p1, p2]) > alpha, or if the
%confidence interval falls outside of the equivalence interval

%INPUTS:
%sample1 and sample2  are the two samples to be compared
%d1:  the lower limit of the equivalence interval
%d2:  the upper limit of the equivalence interval
%alpha:  level of significance (default 0.05).  Resulting confidence
%interval is a (1-2*alpha)% confidence interval

%OUTPUTS:
%p1:  the p value associated with the probability that M1-M2 falls to the 
%left of d1
%p2:  the p value associated with the likelihood that M1-M2 falls to the
%right of d2
%CI:  Confidence interval, (1-2*alpha)%.  Default is 90%

M1 = mean(sample1); %mean of distribution 1
M2 = mean(sample2); %mean of distribution 2

n1 = length(sample1); n2 = length(sample2); %distribution sample sizes
s1 = std(sample1); s2 = std(sample2); %standard deviations of the distributions

SEM = ( ( ( (n1-1).*s1^2+(n2-1).*s2^2 )./(n1+n2-2) ).* (1/n1 + 1/n2)  ).^(1/2);

t1 = ((M1-M2)-d1)/SEM;
t2 = ((M1-M2)-d2)/SEM;

p1 = 1-tcdf(t1,n1+n2-2);
p2 = tcdf(t2,n1+n2-2);

if isempty(alpha)
    alpha = 0.05;
end

zcrit = abs(norminv(alpha,0,1));
CI = [(M1-M2) - zcrit*SEM, (M1-M2) + zcrit*SEM];
