#include <iostream>
#include <string.h>

#include "icpPointToPlane.h"
#include "icpPointToPoint.h"

using namespace std;

// Tr: output SE3 transformation
// model: model points
// temp: template points
// T: initial transformation
// indist: inlier distance
// num_model: model point number
// num_temp: template point number
// dim: point dimension
// flag: use point-to-point or point-to-plane
void _icp(double* Tr, double* model, double* temp, double* T, double indist, int num_model, int num_temp, int dim, int flag)
{
  // input initial transformation (dim=2)
  Matrix R, t;
  if (dim == 2) 
  {
    R = Matrix(2, 2);
    t = Matrix(2, 1);
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 2; j++)
      {
        R.val[i][j] = T[i * 3 + j];
      }
      t.val[i][0] = T[i * 3 + 2];
    }
  // input initial transformation (dim=3)
  }
  else 
  {
    R = Matrix(3, 3);
    t = Matrix(3, 1);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        R.val[i][j] = T[i * 4 + j];
      }
      t.val[i][0] = T[i * 4 + 3];
    }
  }
  
  // run icp
  if (flag == 0)  // point-to-point
  {
    IcpPointToPoint icp(model, num_model, dim);
    icp.fit(temp, num_temp, R, t, indist);
  }
  else  // point-to-plane
  {
    IcpPointToPlane icp(model, num_model, dim);
    icp.fit(temp, num_temp, R, t, indist);
  }
  
  // output final transformation (dim=2)
  if (dim == 2) 
  {
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 2; j++)
      {
        Tr[i * 3 + j] = R.val[i][j];
      }
      Tr[i * 3 + 2] = t.val[i][0];
    }
    Tr[8] = 1;
  // output final transformation (dim=3)
  }
  else 
  {    
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        Tr[i * 4 + j] = R.val[i][j];
      }
      Tr[i * 4 + 3] = t.val[i][0];
    }
    Tr[15] = 1;
  }
}
