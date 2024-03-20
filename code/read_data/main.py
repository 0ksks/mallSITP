from combine_import_complex import main as main0
from combine_import_plain import main as main1
from import_customer_stream import main as main2
from sql_engine import engine
if __name__ == "__main__":
    main0(engine)
    main1(engine)
    main2(engine)
    engine.dispose()