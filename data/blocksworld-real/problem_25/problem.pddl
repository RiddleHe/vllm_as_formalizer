(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        green - block
        red - block
        yellow - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on blue orange)
        (clear blue)
        (ontable green)
        (on red green)
        (on yellow red)
        (clear yellow)
        (ontable purple)
        (clear purple)
        ; orange -> blue
        ; green -> red -> yellow
        ; purple
    )
    (:goal (and (on red yellow)
        (on orange red)
        (on purple green)
        (on blue purple)))
)
    