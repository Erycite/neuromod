mv IPSPsearch IPSPsearch_old
mv EPSPsearch EPSPsearch_old
mv IPSP_search_map.svg IPSP_search_map_old.svg
mv EPSP_search_map.svg EPSP_search_map_old.svg

rm -R IPSPsearch
rm -R EPSPsearch

python run.py --folder IPSPsearch --params ipsp_response.py --search search.py --map yes nest
python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
python plot_map.py
