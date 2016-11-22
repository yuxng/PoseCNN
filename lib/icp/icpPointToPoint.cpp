/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

Authors: Andreas Geiger

openMP support by Manolis Lourakis, Foundation for Research & Technology - Hellas, Heraklion, Greece

libicp is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libicp is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libicp; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "icpPointToPoint.h"

using namespace std;

// Also see (3d part): "Least-Squares Fitting of Two 3-D Point Sets" (Arun, Huang and Blostein)
double IcpPointToPoint::fitStep (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active) {
  
  int i;
  int nact = (int)active.size();

  // init matrix for point correspondences
  Matrix p_m(nact,dim); // model
  Matrix p_t(nact,dim); // template
  
  // init mean
  Matrix mu_m(1,dim);
  Matrix mu_t(1,dim);
  
  // dimensionality 2
  if (dim==2) {
    
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1];
    double r10 = R.val[1][0]; double r11 = R.val[1][1];
    double t0  = t.val[0][0]; double t1  = t.val[1][0];
    double mum0 = 0.0, mum1 = 0.0;
    double mut0 = 0.0, mut1 = 0.0;

    // establish correspondences
#pragma omp parallel for private(i) default(none) shared(T,active,nact,p_m,p_t,r00,r01,r10,r11,t0,t1) reduction(+:mum0,mum1, mut0,mut1) // schedule (dynamic,2)
    for (i=0; i<nact; i++) {
      // kd tree query + result
      std::vector<float>         query(dim);
      kdtree::KDTreeResultVector result;
  
      // get index of active point
      int32_t idx = active[i];

      // transform point according to R|t
      query[0] = (float)(r00*T[idx*2+0] + r01*T[idx*2+1] + t0);
      query[1] = (float)(r10*T[idx*2+0] + r11*T[idx*2+1] + t1);

      // search nearest neighbor
      M_tree->n_nearest(query,1,result);

      // set model point
      p_m.val[i][0] = M_tree->the_data[result[0].idx][0]; mum0 += p_m.val[i][0];
      p_m.val[i][1] = M_tree->the_data[result[0].idx][1]; mum1 += p_m.val[i][1];

      // set template point
      p_t.val[i][0] = query[0]; mut0 += p_t.val[i][0];
      p_t.val[i][1] = query[1]; mut1 += p_t.val[i][1];
    }
    mu_m.val[0][0] = mum0;
    mu_m.val[0][1] = mum1;

    mu_t.val[0][0] = mut0;
    mu_t.val[0][1] = mut1;
    
  // dimensionality 3
  } else {
    
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1]; double r02 = R.val[0][2];
    double r10 = R.val[1][0]; double r11 = R.val[1][1]; double r12 = R.val[1][2];
    double r20 = R.val[2][0]; double r21 = R.val[2][1]; double r22 = R.val[2][2];
    double t0  = t.val[0][0]; double t1  = t.val[1][0]; double t2  = t.val[2][0];
    double mum0 = 0.0, mum1 = 0.0, mum2 = 0.0;
    double mut0 = 0.0, mut1 = 0.0, mut2 = 0.0;

    // establish correspondences
#pragma omp parallel for private(i) default(none) shared(T,active,nact,p_m,p_t,r00,r01,r02,r10,r11,r12,r20,r21,r22,t0,t1,t2) reduction(+:mum0,mum1,mum2, mut0,mut1,mut2) // schedule (dynamic,2)
    for (i=0; i<nact; i++) {
      // kd tree query + result
      std::vector<float>         query(dim);
      kdtree::KDTreeResultVector result;

      // get index of active point
      int32_t idx = active[i];

      // transform point according to R|t
      query[0] = (float)(r00*T[idx*3+0] + r01*T[idx*3+1] + r02*T[idx*3+2] + t0);
      query[1] = (float)(r10*T[idx*3+0] + r11*T[idx*3+1] + r12*T[idx*3+2] + t1);
      query[2] = (float)(r20*T[idx*3+0] + r21*T[idx*3+1] + r22*T[idx*3+2] + t2);

      // search nearest neighbor
      M_tree->n_nearest(query,1,result);

      // set model point
      p_m.val[i][0] = M_tree->the_data[result[0].idx][0]; mum0 += p_m.val[i][0];
      p_m.val[i][1] = M_tree->the_data[result[0].idx][1]; mum1 += p_m.val[i][1];
      p_m.val[i][2] = M_tree->the_data[result[0].idx][2]; mum2 += p_m.val[i][2];

      // set template point
      p_t.val[i][0] = query[0]; mut0 += p_t.val[i][0];
      p_t.val[i][1] = query[1]; mut1 += p_t.val[i][1];
      p_t.val[i][2] = query[2]; mut2 += p_t.val[i][2];
    }
    mu_m.val[0][0] = mum0;
    mu_m.val[0][1] = mum1;
    mu_m.val[0][2] = mum2;

    mu_t.val[0][0] = mut0;
    mu_t.val[0][1] = mut1;
    mu_t.val[0][2] = mut2;
  }
  
  // subtract mean
  mu_m = mu_m/(double)active.size();
  mu_t = mu_t/(double)active.size();
  Matrix q_m = p_m - Matrix::ones(active.size(),1)*mu_m;
  Matrix q_t = p_t - Matrix::ones(active.size(),1)*mu_t;

  // compute relative rotation matrix R and translation vector t
  Matrix H = ~q_t*q_m;
  Matrix U,W,V;
  H.svd(U,W,V);
  Matrix R_ = V*~U;

  // fix improper matrix problem
  if (R_.det()<0){
    Matrix B = Matrix::eye(dim);
    B.val[dim-1][dim-1] = R_.det();
    R_ = V*B*~U;
  }

  Matrix t_ = ~mu_m - R_*~mu_t;
  
  // compose: R|t = R_|t_ * R|t
  R = R_*R;
  t = R_*t+t_;

  // return max delta in parameters
  if (dim==2) return max((R_-Matrix::eye(2)).l2norm(),t_.l2norm());
  else        return max((R_-Matrix::eye(3)).l2norm(),t_.l2norm());
}

std::vector<int32_t> IcpPointToPoint::getInliers (double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const double indist) {

  // init inlier vector + query point + query result
  vector<int32_t>            inliers;
  std::vector<float>         query(dim);
  kdtree::KDTreeResultVector neighbor;
  
  // dimensionality 2
  if (dim==2) {
  
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1];
    double r10 = R.val[1][0]; double r11 = R.val[1][1];
    double t0  = t.val[0][0]; double t1  = t.val[1][0];

    // check for all points if they are inliers
    for (int32_t i=0; i<T_num; i++) {

      // transform point according to R|t
      query[0] = (float)(r00*T[i*2+0] + r01*T[i*2+1] + t0);
      query[1] = (float)(r10*T[i*2+0] + r11*T[i*2+1] + t1);

      // search nearest neighbor
      M_tree->n_nearest(query,1,neighbor);

      // check if it is an inlier
      if (neighbor[0].dis<indist)
        inliers.push_back(i);
    }
    
  // dimensionality 3
  } else {
    
    // extract matrix and translation vector
    double r00 = R.val[0][0]; double r01 = R.val[0][1]; double r02 = R.val[0][2];
    double r10 = R.val[1][0]; double r11 = R.val[1][1]; double r12 = R.val[1][2];
    double r20 = R.val[2][0]; double r21 = R.val[2][1]; double r22 = R.val[2][2];
    double t0  = t.val[0][0]; double t1  = t.val[1][0]; double t2  = t.val[2][0];

    // check for all points if they are inliers
    for (int32_t i=0; i<T_num; i++) {

      // transform point according to R|t
      query[0] = (float)(r00*T[i*3+0] + r01*T[i*3+1] + r02*T[i*3+2] + t0);
      query[1] = (float)(r10*T[i*3+0] + r11*T[i*3+1] + r12*T[i*3+2] + t1);
      query[2] = (float)(r20*T[i*3+0] + r21*T[i*3+1] + r22*T[i*3+2] + t2);

      // search nearest neighbor
      M_tree->n_nearest(query,1,neighbor);

      // check if it is an inlier
      if (neighbor[0].dis<indist)
        inliers.push_back(i);
    }
  }
  
  // return vector with inliers
  return inliers;
}
