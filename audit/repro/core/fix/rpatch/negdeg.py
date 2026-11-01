import re
def load(p):
    raw=open(p,'rb').read().decode('utf-8'); return raw.replace('\r\n','\n'), ('\r\n' in raw)
def save(p,t,crlf):
    if crlf: t=t.replace('\n','\r\n')
    open(p,'wb').write(t.encode('utf-8'))
# helper in partial_moments.h
p='partial_moments.h'; t,c=load(p)
old='''namespace nns_pm_detail {
'''
new='''// Partial moments are defined for degree >= 0. A negative degree used to be
// evaluated silently (repeat-multiplication with n < 0 returns 1, so the
// integer fast path returned the degree-0 count), and NaN reached
// static_cast<int>(NaN). Called from the R-facing wrappers before any
// parallel work starts.
inline void nns_check_degree(const double degree, const char *name) {
  if (!(degree >= 0.0)) {
    Rcpp::stop("'%s' must be a non-negative number", name);
  }
}

namespace nns_pm_detail {
'''
assert t.count(old)==1; t=t.replace(old,new,1); save(p,t,c)

p='partial_moments_rcpp.cpp'; t,c=load(p)
def ins(sig, checks):
    global t
    i=t.index(sig); j=t.index('{\n',i)+2
    t=t[:j]+checks+t[j:]
ins('double CoLPM_nD_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins('double CoUPM_nD_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins('double DPM_nD_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins('NumericVector LPM_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins('NumericVector UPM_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins('NumericVector LPM_ratio_RCPP(', '   nns_check_degree(degree, "degree");\n')
ins('NumericVector UPM_ratio_RCPP(', '   nns_check_degree(degree, "degree");\n')
ins('NumericVector CoLPM_RCPP(', '   nns_check_degree(degree_lpm, "degree_lpm");\n   nns_check_degree(degree_y, "degree_y");\n')
ins('NumericVector CoUPM_RCPP(', '   nns_check_degree(degree_upm, "degree_upm");\n   nns_check_degree(degree_y, "degree_y");\n')
ins('NumericVector DLPM_RCPP(', '   nns_check_degree(degree_lpm, "degree_lpm");\n   nns_check_degree(degree_upm, "degree_upm");\n')
ins('NumericVector DUPM_RCPP(', '   nns_check_degree(degree_lpm, "degree_lpm");\n   nns_check_degree(degree_upm, "degree_upm");\n')
ins('List PMMatrix_RCPP(', '   nns_check_degree(LPM_degree, "LPM_degree");\n   nns_check_degree(UPM_degree, "UPM_degree");\n')
save(p,t,c)

p='partial_moments.cpp'; t,c=load(p)
def ins2(sig, checks):
    global t
    i=t.index(sig); j=t.index('{\n',i)+2
    t=t[:j]+checks+t[j:]
ins2('NumericVector CoLPM_nD_batch_RCPP(', '  nns_check_degree(degree, "degree");\n')
ins2('List PMMatrix_CPv(', '  nns_check_degree(LPM_degree, "LPM_degree");\n  nns_check_degree(UPM_degree, "UPM_degree");\n')
save(p,t,c)
