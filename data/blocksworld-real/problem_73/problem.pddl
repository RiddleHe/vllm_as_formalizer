(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        green - block
        red - block
        orange - block
        blue - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on green yellow)
        (on red green)
        (clear red)
        (ontable orange)
        (on blue orange)
        (clear blue)
        (ontable purple)
        (clear purple)
        ; yellow -> green -> red
        ; orange -> blue
        ; purple
    )
    (:goal (and (on orange blue)
        (on yellow orange)
        (on red yellow)
        (on green purple)))
)
    