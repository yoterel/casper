// see "Image Deformation Using Moving Least Square "
#ifndef MLS_H
#define MLS_H
#include <vector>
#include <opencv2/opencv.hpp>

typedef struct _typeA
{
	cv::Mat a;
	cv::Mat b;
	cv::Mat c;
	cv::Mat d;
} typeA;

typedef struct _typeRigid
{
	std::vector<_typeA> A;
	cv::Mat normof_v_Pstar;
} typeRigid;

/*Function to Compute the Weights*/
inline cv::Mat MLSprecomputeWeights(cv::Mat p, cv::Mat v, double a)
{
	cv::Mat w = cv::Mat::zeros(p.cols, v.cols, CV_32F);
	cv::Mat p_resize;
	cv::Mat norms = cv::Mat::zeros(2, v.cols, CV_32F);
	cv::Mat norms_a;
	cv::Mat p_v;

	// Iterate through the control points
	for (int i = 0; i < p.cols; i++)
	{
		// compute the norms
		p_resize = repeat(p.col(i), 1, v.cols);
		p_v = p_resize - v;
		pow(p_v, 2, p_v);
		norms = p_v.row(0) + p_v.row(1);
		pow(norms, a, norms_a);
		// compute the weights
		w.row(i) = 1.0 / norms_a;
	}
	return w;
}

/*Function to Precompute Weighted Centroids*/
inline cv::Mat MLSprecomputeWCentroids(cv::Mat p, cv::Mat w)
{
	cv::Mat Pstar;
	cv::Mat mult;
	cv::Mat resize;
	cv::Mat sum = cv::Mat::zeros(1, w.cols, CV_32F);
	mult = p * w;

	for (int i = 0; i < w.rows; i++)
		sum += w.row(i);
	resize = repeat(sum, p.rows, 1);

	Pstar = mult / resize;
	return Pstar;
}

/*Function to Precompute Affine Deformation*/
inline cv::Mat MLSprecomputeAffine(cv::Mat p, cv::Mat v, cv::Mat w)
{
	// Precompute Weighted Centroids
	cv::Mat Pstar = MLSprecomputeWCentroids(p, w);
	cv::Mat M1;
	cv::Mat a = cv::Mat::zeros(1, Pstar.cols, CV_32F);
	cv::Mat b = cv::Mat::zeros(1, Pstar.cols, CV_32F);
	cv::Mat d = cv::Mat::zeros(1, Pstar.cols, CV_32F);
	cv::Mat row1, row2, power_b;
	cv::Mat det;
	// Precompute the first matrix
	M1 = v - Pstar;
	std::vector<cv::Mat> Phat;
	// Iterate through control points
	for (int i = 0; i < p.cols; i++)
	{
		cv::Mat t = repeat(p.col(i), 1, Pstar.cols) - Pstar;

		pow(t.row(0), 2, row1);
		pow(t.row(1), 2, row2);
		// compute matrix elements
		a = a + (w.row(i)).mul(row1);
		b = b + ((w.row(i)).mul(t.row(0))).mul(t.row(1));
		d = d + row2.mul(w.row(i));
		// save the hat points
		Phat.push_back(t);
	}

	pow(b, 2, power_b);
	// compute determinant
	det = a.mul(d) - power_b;
	// compute the inverse
	cv::Mat Ia = d / det;
	cv::Mat Ib = -1 * b / det;
	cv::Mat Id = a / det;

	cv::Mat Iab, Ibd;
	vconcat(Ia, Ib, Iab);
	vconcat(Ib, Id, Ibd);

	cv::Mat m = M1.mul(Iab);
	cv::Mat sum1 = cv::Mat::zeros(1, m.cols, CV_32F);
	for (int i = 0; i < m.rows; i++)
		sum1 += m.row(i);

	cv::Mat n = M1.mul(Ibd);
	cv::Mat sum2 = cv::Mat::zeros(1, n.cols, CV_32F);
	for (int i = 0; i < n.rows; i++)
		sum2 += n.row(i);

	// compute the first product element
	cv::Mat F1;
	vconcat(sum1, sum2, F1);

	cv::Mat A = cv::Mat::zeros(p.cols, Pstar.cols, CV_32F);
	cv::Mat F1_mul;

	for (int j = 0; j < p.cols; j++)
	{
		F1_mul = F1.mul(Phat.at(j));
		cv::Mat temp = cv::Mat::zeros(1, F1_mul.cols, CV_32F);
		for (int i = 0; i < F1_mul.rows; i++)
			temp += F1_mul.row(i);
		A.row(j) = temp.mul(w.row(j));
	}
	cv::Mat data = A;

	return data;
}

// precompute Asimilar
inline std::vector<_typeA> MLSprecomputeA(cv::Mat Pstar, std::vector<cv::Mat> Phat, cv::Mat v, cv::Mat w)
{
	std::vector<_typeA> A;

	// fixed part
	cv::Mat R1 = v - Pstar;
	cv::Mat R2;
	vconcat(R1.row(1), -R1.row(0), R2);

	for (int i = 0; i < Phat.size(); i++)
	{
		// precompute
		typeA temp;
		cv::Mat L1 = Phat.at(i);
		cv::Mat L2;
		vconcat(L1.row(1), (L1.row(0)).mul(-1), L2);

		cv::Mat L1R1 = L1.mul(R1);
		cv::Mat sumL1R1 = cv::Mat::zeros(1, L1R1.cols, CV_32F);

		cv::Mat L1R2 = L1.mul(R2);
		cv::Mat sumL1R2 = cv::Mat::zeros(1, L1R2.cols, CV_32F);

		cv::Mat L2R1 = L2.mul(R1);
		cv::Mat sumL2R1 = cv::Mat::zeros(1, L2R1.cols, CV_32F);

		cv::Mat L2R2 = L2.mul(R2);
		cv::Mat sumL2R2 = cv::Mat::zeros(1, L2R2.cols, CV_32F);

		for (int j = 0; j < L1R1.rows; j++)
			sumL1R1 += L1R1.row(j);

		for (int j = 0; j < L1R2.rows; j++)
			sumL1R2 += L1R2.row(j);

		for (int j = 0; j < L2R1.rows; j++)
			sumL2R1 += L2R1.row(j);

		for (int j = 0; j < L2R2.rows; j++)
			sumL2R2 += L2R2.row(j);

		temp.a = (w.row(i)).mul(sumL1R1);
		temp.b = (w.row(i)).mul(sumL1R2);
		temp.c = (w.row(i)).mul(sumL2R1);
		temp.d = (w.row(i)).mul(sumL2R2);

		A.push_back(temp);
	}

	return A;
}

/* \frac{1}{\mu_s}A_i */
inline std::vector<_typeA> MLSprecomputeSimilar(cv::Mat p, cv::Mat v, cv::Mat w)
{
	cv::Mat Pstar = MLSprecomputeWCentroids(p, w);
	std::vector<cv::Mat> Phat;
	cv::Mat mu = cv::Mat::zeros(1, Pstar.cols, CV_32F);
	cv::Mat t1;
	cv::Mat product;
	for (int i = 0; i < p.cols; i++)
	{
		cv::Mat t = repeat(p.col(i), 1, Pstar.cols) - Pstar;
		cv::Mat sum = cv::Mat::zeros(1, t.cols, CV_32F);
		pow(t, 2, t1);
		for (int j = 0; j < t1.rows; j++)
			sum += t1.row(j);

		mu = mu + (w.row(i)).mul(sum);
		Phat.push_back(t);
	}

	std::vector<_typeA> A = MLSprecomputeA(Pstar, Phat, v, w);

	for (int i = 0; i < A.size(); i++)
	{
		(A.at(i)).a = ((A.at(i)).a).mul(1 / mu);
		(A.at(i)).b = ((A.at(i)).b).mul(1 / mu);
		(A.at(i)).c = ((A.at(i)).c).mul(1 / mu);
		(A.at(i)).d = ((A.at(i)).d).mul(1 / mu);
	}
	return A;
}

/* f_r(v)-> */
inline _typeRigid MLSprecomputeRigid(cv::Mat p, cv::Mat v, cv::Mat w)
{
	typeRigid data;
	cv::Mat Pstar = MLSprecomputeWCentroids(p, w);
	std::vector<cv::Mat> Phat;
	for (int i = 0; i < p.cols; i++)
	{
		cv::Mat t = repeat(p.col(i), 1, Pstar.cols) - Pstar;
		Phat.push_back(t);
	}

	std::vector<_typeA> A = MLSprecomputeA(Pstar, Phat, v, w); // �õ�A
	cv::Mat v_Pstar = v - Pstar;
	cv::Mat vpower;
	pow(v_Pstar, 2, vpower);
	cv::Mat sum = cv::Mat::zeros(1, vpower.cols, CV_32F);
	for (int i = 0; i < vpower.rows; i++)
		sum += vpower.row(i);

	sqrt(sum, data.normof_v_Pstar);
	data.A = A;
	return data;
}

/* f_r(v) */
inline cv::Mat MLSPointsTransformRigid(cv::Mat w, _typeRigid mlsd, cv::Mat q)
{
	cv::Mat Qstar = MLSprecomputeWCentroids(q, w);
	cv::Mat Qhat;
	cv::Mat fv2 = cv::Mat::zeros(Qstar.rows, Qstar.cols, CV_32F);
	cv::Mat prod1, prod2;
	cv::Mat con1, con2;
	cv::Mat update;
	cv::Mat repmat;
	cv::Mat npower;
	cv::Mat normof_fv2;
	cv::Mat fv = cv::Mat::zeros(Qstar.rows, Qstar.cols, CV_32F);
	for (int i = 0; i < q.cols; i++)
	{
		Qhat = repeat(q.col(i), 1, Qstar.cols) - Qstar;

		vconcat((mlsd.A.at(i)).a, (mlsd.A.at(i)).c, con1);
		prod1 = Qhat.mul(con1);
		cv::Mat sum1 = cv::Mat::zeros(1, prod1.cols, CV_32F);
		for (int j = 0; j < prod1.rows; j++)
			sum1 += prod1.row(j);

		vconcat((mlsd.A.at(i)).b, (mlsd.A.at(i)).d, con2);
		prod2 = Qhat.mul(con2);
		cv::Mat sum2 = cv::Mat::zeros(1, prod2.cols, CV_32F);
		for (int j = 0; j < prod2.rows; j++)
			sum2 += prod2.row(j);

		vconcat(sum1, sum2, update);
		fv2 = fv2 + update;
	}
	npower = fv2.mul(fv2);

	cv::Mat sumfv2 = cv::Mat::zeros(1, npower.cols, CV_32F);
	for (int i = 0; i < npower.rows; i++)
		sumfv2 += npower.row(i);

	sqrt(sumfv2, normof_fv2);

	cv::Mat norm_fact = (mlsd.normof_v_Pstar).mul(1 / normof_fv2);

	repmat = repeat(norm_fact, fv2.rows, 1);
	fv = fv2.mul(repmat) + Qstar;

	return fv;
}

/* f_s(v)*/
inline cv::Mat MLSPointsTransformSimilar(cv::Mat w, std::vector<_typeA> A, cv::Mat q)
{
	cv::Mat Qstar = MLSprecomputeWCentroids(q, w);

	cv::Mat fv = Qstar.clone();
	cv::Mat Qhat;
	cv::Mat resize;

	cv::Mat prod1, prod2;
	cv::Mat con1, con2;

	cv::Mat update;

	for (int i = 0; i < q.cols; i++)
	{
		Qhat = repeat(q.col(i), 1, Qstar.cols) - Qstar;
		vconcat((A.at(i)).a, (A.at(i)).c, con1);
		prod1 = Qhat.mul(con1);
		cv::Mat sum1 = cv::Mat::zeros(1, prod1.cols, CV_32F);
		for (int j = 0; j < prod1.rows; j++)
			sum1 += prod1.row(j);

		vconcat((A.at(i)).b, (A.at(i)).d, con2);
		prod2 = Qhat.mul(con2);
		cv::Mat sum2 = cv::Mat::zeros(1, prod2.cols, CV_32F);
		for (int j = 0; j < prod2.rows; j++)
			sum2 += prod2.row(j);

		vconcat(sum1, sum2, update);
		fv = fv + update;
	}
	return fv;
}

/* f_a(v)  */
inline cv::Mat MLSPointsTransformAffine(cv::Mat w, cv::Mat A, cv::Mat q)
{
	// compute weighted centroids for q
	cv::Mat Qstar = MLSprecomputeWCentroids(q, w);

	cv::Mat fv = Qstar.clone();
	cv::Mat Qhat;
	cv::Mat resize;
	// add the affine parts
	for (int j = 0; j < q.cols; j++)
	{
		// compute hat points
		Qhat = repeat(q.col(j), 1, Qstar.cols) - Qstar;
		resize = repeat(A.row(j), Qhat.rows, 1);
		// update
		fv = fv + Qhat.mul(resize);
	}
	return fv;
}
#endif MLS_H