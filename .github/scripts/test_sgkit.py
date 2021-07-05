import sgkit as sg

if __name__ == "__main__":
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, n_contig=23)
    print(ds)
