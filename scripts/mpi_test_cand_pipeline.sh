#!/bin/bash


mpirun --oversubscribe -np 36 beam_cand_sender : -np 37 mpi_cand_pipeline

