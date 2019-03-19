#!/bin/bash

NUM_TRAIN=256
NUM_TEST=64

# Generate Labyrinth scenarios
echo ""
echo "####################################"
echo "## Generating: Labyrinth Scenario ##"
echo "####################################"
echo ""

for SIZE in $(seq 5 2 13)
do
    echo "Creating scenario of size $SIZE"
    BASE_DIR=scenarios/custom_scenarios/labyrinth/$SIZE/
    mkdir -p $BASE_DIR/train
    mkdir -p $BASE_DIR/test
    python generate_scenario.py --scenario labyrinth --num_train $NUM_TRAIN --num_test $NUM_TEST --size $SIZE --grid_size 128 --scenario_dir $BASE_DIR
done

# Generate Find and return scenarios
echo ""
echo "######################################"
echo "## Generating: Find return Scenario ##"
echo "######################################"
echo ""

for SIZE in $(seq 5 2 13)
do
    echo "Creating scenario of size $SIZE"
    BASE_DIR=scenarios/custom_scenarios/find_return/$SIZE/
    mkdir -p $BASE_DIR/train
    mkdir -p $BASE_DIR/test
    python generate_scenario.py --scenario find_return --num_train $NUM_TRAIN --num_test $NUM_TEST --size $SIZE --grid_size 128 --scenario_dir $BASE_DIR
done

# Generate k-item
echo ""
echo "####################################"
echo "##  Generating: K-item Scenario   ##"
echo "####################################"
echo ""

for ITEM in $(seq 2 2 8)
do
    echo "Creating scenario with $ITEM items"
    BASE_DIR=scenarios/custom_scenarios/kitem/$ITEM/
    mkdir -p $BASE_DIR/train
    mkdir -p $BASE_DIR/test
    python generate_scenario.py --scenario k_item --num_train $NUM_TRAIN --num_test $NUM_TEST --difficulty $ITEM --grid_size 160  --scenario_dir $BASE_DIR
done

# Generate two color
echo ""
echo "####################################"
echo "## Generating: Two color Scenario ##"
echo "####################################"
echo ""

for PROB in $(seq 1 2 7)
do
    echo "Creating scenario with difficulty $PROB"
    BASE_DIR=scenarios/custom_scenarios/two_color/$PROB/
    mkdir -p $BASE_DIR/train
    mkdir -p $BASE_DIR/test

    python generate_scenario.py --scenario two_color --num_train $NUM_TRAIN --num_test $NUM_TEST --difficulty $PROB --grid_size 256 --scenario_dir $BASE_DIR
done


