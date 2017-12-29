package com.sparkMlib.movierecommand1;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by renwujie on 2017/12/29 at 12:46
 *
 * Reference:
 *  1. https://zhuanlan.zhihu.com/p/30035244
 *  2. http://blog.csdn.net/dulinanaaa/article/details/54970962
 *  3. http://blog.csdn.net/qq1010885678/article/details/46052055
 *
 * 采用更新version时会报错：
 *  Caused by: java.lang.NoSuchMethodError: scala.Predef$.$conforms(Lscala/Predef$$less$colon$less;
 *
 * 原因不明，还没解决。
 *
 */
public class MovieRecommand {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("MovieRecommand")
                .setMaster("local");

        JavaSparkContext sc = new JavaSparkContext(conf);

        //load data
        String path = "T:\\ratings.data";
        JavaRDD<String> src = sc.textFile(path);

        // 所有评分数据，由于此数据要分三部分使用，60%用于训练，20%用于验证，最后20%用于测试。将时间戳%10可以得到近似的10等分，用于三部分数据切分
        JavaRDD<Tuple2<Integer, Rating>> srcRDD = src.map(
                new Function<String, Tuple2<Integer, Rating>>() {
                    @Override
                    public Tuple2<Integer, Rating> call(String v) throws Exception {
                        String[] field = v.split("\t");
                        if (field.length != 4) {
                            throw new IllegalArgumentException("Each line must contain 4 fields");
                        }
                        int userId = Integer.parseInt(field[0]);
                        int movieId = Integer.parseInt(field[1]);
                        double rating = Float.valueOf(field[2]);
                        int timetamp = (int) (Long.parseLong(field[3]) % 10);
                        return new Tuple2<Integer, Rating>(timetamp, new Rating(userId, movieId, rating));
                    }
                });

        System.out.println("get" + srcRDD.count() + " ratings from " + srcRDD.distinct().count() + "users on " + srcRDD.distinct().count() + "movies");

        //我的评分数据
        String myPath = "T:\\my.data";
        JavaRDD<String> myData = sc.textFile(myPath);
        JavaRDD<Rating> myDataRDD = myData.map(
                new Function<String, Rating>() {
                    @Override
                    public Rating call(String v) throws Exception {
                        String[] field = v.split("\t");
                        if (field.length != 4) {
                            throw new IllegalArgumentException("Each line must contain 4 fields");
                        }
                        int userId = Integer.parseInt(field[0]);
                        int movieId = Integer.parseInt(field[1]);
                        double rating = Float.valueOf(field[2]);
                        //这里将timespan这列对10做取余操作，这样一来个评分数据的这一列都是一个0-9的数字，做什么用？可以直接用来给数据分区
                        int timetamp = (int) (Long.parseLong(field[3]) % 10);
                        return new Rating(userId, movieId, rating);
                    }
                });


        //分区数
        int numPartitions = 3;

        //将 (timestamp % 10) < 6（60%）的数据用于训练
        JavaRDD<Rating> trainRDD = JavaPairRDD.fromJavaRDD(srcRDD.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Integer, Rating> v) throws Exception {
                        return v._1 < 6;
                    }
                })
        ).values().union(myDataRDD).repartition(numPartitions).cache();


        //将 6 < (timestamp % 10) < 8（20%）的数据用于验证
        JavaRDD<Rating> verifyRDD = JavaPairRDD.fromJavaRDD(srcRDD.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Integer, Rating> v) throws Exception {
                        return v._1 > 6 && v._1 < 8;
                    }
                })
        ).values().repartition(numPartitions).cache();

        //将 (timestamp % 10) > 8（20%）的数据用于测试
        JavaRDD<Rating> testRDD = JavaPairRDD.fromJavaRDD(srcRDD.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Integer, Rating> v) throws Exception {
                        return v._1 >= 8;
                    }
                }
        )).values().cache();

        System.out.println("training data's num : " + trainRDD.count() + " validate data's num : " + verifyRDD.count() + " test data's num : " + testRDD.count());

        // 为 训练集 设置参数 每种参数设置2个值，三层for循环，一共进行8次训练
        //一个三层嵌套循环，会产生8个ranks ，lambdas ，iters 的组合，每个组合都会产生一个模型，计算8个模型的方差，最小的那个记为最佳模型
        List<Integer> ranks = new ArrayList<Integer>();
        ranks.add(8);
        ranks.add(22);

        List<Integer> iters = new ArrayList<Integer>();
        iters.add(5);
        iters.add(7);

        List<Double> lambdas = new ArrayList<Double>();
        lambdas.add(0.1);
        lambdas.add(10.0);


        //初始化最好的模型参数
        MatrixFactorizationModel bestModel = null;
        double bestValidateRnse = Double.MAX_VALUE;
        int bestRank = 0;
        double bestLambda = -1.0;
        int bestIter = -1;

        for (int i = 0; i < ranks.size(); i++) {
            for (int j = 0; j < lambdas.size(); j++) {
                for (int k = 0; k < iters.size(); k++) {
                    //训练获得模型
                    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainRDD), ranks.get(i), iters.get(j), lambdas.get(k));
                    //通过校验集validateData_Rating获取方差，以便查看此模型的好坏
                    double validateRnse = variance(model, verifyRDD, verifyRDD.count());

                    System.out.println("validation = " + validateRnse + " for the model trained with rank = " + ranks.get(i) + " lambda = " + lambdas.get(k) + " and numIter" + iters.get(j));

                    //将最好的模型训练结果所设置的参数进行保存
                    if (validateRnse < bestValidateRnse) {

                    }
                }
            }
        }

        //8次训练后获取最好的模型，根据最好的模型及训练集testRDD来获取此方差
        double testDataRnse = variance(bestModel, testRDD, testRDD.count());
        System.out.println("the best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
                + " and numIter = " + bestIter + " and Rnse on the test data is " + testDataRnse);

        // 获取测试数据中，分数的平均值
        final double meanRating = trainRDD.union(verifyRDD).mapToDouble(
                new DoubleFunction<Rating>() {
                    @Override
                    public double call(Rating r) throws Exception {
                        return r.rating();
                    }
                }).mean();

        // 根据平均值来计算旧的方差值
        double baseLineRnse = Math.sqrt(testRDD.mapToDouble(
                new DoubleFunction<Rating>() {
                    @Override
                    public double call(Rating r) throws Exception {
                        return (meanRating - r.rating()) * (meanRating - r.rating());
                    }
                }).mean());

        // 通过模型，数据的拟合度提升了多少
        double improvment = (baseLineRnse - testDataRnse) / baseLineRnse * 100;
        System.out.println("the best model improves the baseline by " + improvment + "%");

        //加载电影数据
        String moviePath = "T:\\movies.data";
        JavaRDD<String> movieData = sc.textFile(moviePath);

        // 将电影的id，标题，类型以三元组的形式保存
        JavaRDD<Tuple3<Integer, String, String>> movieListRDD = movieData.map(
                new Function<String, Tuple3<Integer, String, String>>() {
                    @Override
                    public Tuple3<Integer, String, String> call(String line) throws Exception {
                        String[] field = line.split("\t");
                        if (field.length != 3) {
                            throw new IllegalArgumentException("Each line must contain 3 fields");
                        }
                        int id = Integer.parseInt(field[0]);
                        String title = field[1];
                        String type = field[2];

                        return new Tuple3<Integer, String, String>(id, title, type);

                    }
                });

        // 将电影的id，标题以二元组的形式保存
        JavaRDD<Tuple2<Integer, String>> movies_Map = movieListRDD.map(
                new Function<Tuple3<Integer, String, String>, Tuple2<Integer, String>>() {

                    @Override
                    public Tuple2<Integer, String> call(Tuple3<Integer, String, String> v) throws Exception {
                        return new Tuple2<Integer, String>(v._1(), v._2());
                    }
                });

        System.out.println("movies recommond for you:");

        // 获取我所看过的电影ids
        final List<Integer> movieIds = myDataRDD.map(new Function<Rating, Integer>() {
            @Override
            public Integer call(Rating r) throws Exception {
                return r.product();
            }
        }).collect();// def collect(): JList[T] --- Return an array that contains all of the elements in this RDD.


        // 从电影数据中去除我看过的电影数据
        JavaRDD<Tuple2<Integer, String>> movieIdList = movies_Map.filter(new Function<Tuple2<Integer, String>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Integer, String> r) throws Exception {
                return movieIds.contains(r._1);
            }
        });

        // 封装rating的参数形式，user为0，product为电影id进行封装
        JavaPairRDD<Integer, Integer> recmmondList = JavaPairRDD.fromJavaRDD(movieIdList.map(
                //public interface Function<T1, R>，第一个参数是传入类型，第二个是传出类型
                new Function<Tuple2<Integer,String>, Tuple2<Integer,Integer>>() {
                    @Override
                    public Tuple2<Integer, Integer> call(Tuple2<Integer, String> t) throws Exception {
                        return new Tuple2<Integer, Integer>(0, t._1);
                    }
                }
        ));

        //通过模型预测出user为0的各pr oduct(电影id)的评分，并按照评分进行排序，获取前10个电影id
        final List<Integer> resultList = bestModel.predict(recmmondList).sortBy(new Function<Rating, Double>() {
            @Override
            public Double call(Rating r) throws Exception {
                return r.rating();
            }
        }, false, 1).map(new Function<Rating, Integer>() {
            @Override
            public Integer call(Rating r) throws Exception {
                return r.product();
            }
        }).take(10);

        //从电影数据中过滤出这10部电影，遍历打印
        if(resultList != null && !resultList.isEmpty()){
            movieListRDD.filter(new Function<Tuple3<Integer, String, String>, Boolean>() {
                @Override
                public Boolean call(Tuple3<Integer, String, String> tuple) throws Exception {
                    return resultList.contains(tuple._1());
                }
            }).foreach(new VoidFunction<Tuple3<Integer, String, String>>() {
                @Override
                public void call(Tuple3<Integer, String, String> t) throws Exception {
                    System.out.println("nmovie name --> " + t._2() + " nmovie type --> " + t._3());
                }
            });

        }
    }

    //方差计算方法
    public static double variance(MatrixFactorizationModel model, JavaRDD<Rating> predictionData, long n) {

        //将predictionData转化成二元组型式，以便训练使用
        JavaRDD<Tuple2<Object, Object>> userProducts = predictionData.map(new Function<Rating, Tuple2<Object, Object>>() {
            @Override
            public Tuple2<Object, Object> call(Rating v) throws Exception {
                return new Tuple2<Object, Object>(v.user(), v.product());
            }
        });

        //通过模型对数据进行预测
        JavaPairRDD<Tuple2<Integer, Integer>, Double> prediction = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            @Override
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating v) throws Exception {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(new Tuple2<>(v.user(), v.product()), v.rating());
                            }
                        })
        );

        //预测值和原值内连接
        JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(predictionData.map(new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
            @Override
            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating v) throws Exception {
                return new Tuple2<Tuple2<Integer, Integer>, Double>(new Tuple2<Integer, Integer>(v.user(), v.product()), v.rating());
            }
        })).join(prediction).values();

        //计算方差并返回结果
        Double dVar = ratesAndPreds.map(new Function<Tuple2<Double, Double>, Double>() {
            @Override
            public Double call(Tuple2<Double, Double> v) throws Exception {
                return (v._1 - v._2) * (v._1 - v._2);
            }
        }).reduce(new Function2<Double, Double, Double>() {
            @Override
            public Double call(Double v1, Double v2) throws Exception {
                return v1 + v2;
            }
        });

        return Math.sqrt(dVar / n);
    }


}
