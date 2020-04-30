enum BC_TYPE { DIRICHLET, NEUMANN };
typedef struct {
  int Cnt;
  int BoCo[6];
  int BoCo_Ref[6];
  float BoCo_Val[6];
} GmlParSct;