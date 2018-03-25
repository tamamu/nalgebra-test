//#[macro_use]
//extern crate approx;
extern crate nalgebra as na;
extern crate num_traits;
extern crate alga;
use num_traits::{One, Zero};
use alga::general::{ClosedAdd, ClosedMul};
use na::*;
use na::storage::*;
use na::constraint::*;

type NumOfFeature = U3;
type NumOfData = U100;
type NumOfOutput = U2;

type InputMatrix  = Matrix<f64, NumOfData, NumOfFeature, MatrixArray<f64, NumOfData, NumOfFeature>>;
type WeightMatrix = Matrix<f64, NumOfFeature, NumOfOutput, MatrixArray<f64, NumOfFeature, NumOfOutput>>;
type ResultMatrix = Matrix<f64, NumOfData, NumOfOutput, MatrixArray<f64, NumOfData, NumOfOutput>>;

fn mul<N, R1: Dim, C1: Dim, SA, R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
    a: &Matrix<N, R1, C1, SA>,
    b: &Matrix<N, R2, C2, SB>,
    out: &mut Matrix<N, R3, C3, SC>
)
where
    N: Scalar + One + Zero + ClosedAdd + ClosedMul,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    SC: StorageMut<N, R3, C3>,
    ShapeConstraint: SameNumberOfRows<R3, R1> + SameNumberOfColumns<C3, C2> + AreMultipliable<R1, C1, R2, C2>
{
    a.mul_to(b, out)
}

fn main() {
    let x: InputMatrix = InputMatrix::new_random();
    let w: WeightMatrix = WeightMatrix::new_random();
    let mut y: ResultMatrix = ResultMatrix::zeros();
    mul(&x, &w, &mut y);

    println!("{:?}", y);
}
